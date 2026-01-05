import random
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import lightning as L
import numpy as np
from matplotlib import pyplot as plt
from pytorch_msssim import SSIM

from .vae import BoxEmbedVAE
from .box_utils import HierarchicalBoxEmbeddingsPrior, BoxEmbeddingDistribution, bessel_volume, soft_box_weighted_intersection, pairwise_gumbel_intersection

class HierarchicalBoxEmbeddingsVAE(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config["model"]["config"]

        self.embed_dim = self.model_config["embed_dim"]
        self.hidden_dims = self.model_config["hidden_dims"]
        self.grid_size = self.model_config["grid_size"]
        self.input_resolution = self.model_config["input_resolution"]
        self.beta_scale = self.model_config["beta_scale"]
        self.crop_padding_range = self.model_config["crop_padding_range"]
        
        self.vae = BoxEmbedVAE(
            latent_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            input_resolution=self.input_resolution,
            beta_scale = self.beta_scale
        )

        self.prior = HierarchicalBoxEmbeddingsPrior(
            prior_config=self.model_config["prior_config"],
            embed_dim=self.embed_dim,
            beta_scale=self.beta_scale,
            # boxes_per_level=self.model_config["prior_config"]["boxes_per_level"],
            # embed_dim=self.embed_dim,
            # beta_scale = self.beta_scale,
            # init_config=self.model_config["prior_config"]["init_config"]
        )

        self.loss_weights = self.model_config["loss_weights"]
        self.ssim_loss_fn = SSIM(
            win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3
        )

        self.viz_datapoint = None

    def crop_to_content(self, images, masks):
        
        cropped_batch = []
        H, W = images.shape[2], images.shape[3]
        
        for i in range(images.shape[0]):
            img = images[i] # (C, H, W)
            mask = masks[i] # (H, W)
            
            non_zero_indices = torch.nonzero(mask)
            
            if non_zero_indices.size(0) == 0:
                cropped_batch.append(img)
                continue
                
            y_min = torch.min(non_zero_indices[:, 0]).item()
            y_max = torch.max(non_zero_indices[:, 0]).item()
            x_min = torch.min(non_zero_indices[:, 1]).item()
            x_max = torch.max(non_zero_indices[:, 1]).item()

            y_min_pad = max(0, y_min - random.randint(self.crop_padding_range[0], self.crop_padding_range[0]))
            y_max_pad = min(H - 1, y_max + random.randint(self.crop_padding_range[0], self.crop_padding_range[0]))
            x_min_pad = max(0, x_min - random.randint(self.crop_padding_range[0], self.crop_padding_range[0]))
            x_max_pad = min(W - 1, x_max + random.randint(self.crop_padding_range[0], self.crop_padding_range[0]))
            
            crop = img[:, y_min_pad : y_max_pad + 1, x_min_pad : x_max_pad + 1]
            
            crop_resized = F.interpolate(
                crop.unsqueeze(0), 
                size=self.input_resolution, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            cropped_batch.append(crop_resized)
            
        return torch.stack(cropped_batch)

    def forward(self, x):
        images = x["images"]
        masks = x["object_masks"]
        
        images = self.crop_to_content(images, masks)

        patches = self.divide_image_into_patches(images) # (B, NumPatches, C, H_p, W_p)
        B, N, C, Hp, Wp = patches.shape
        patch_end_idx = B * N

        patches_flat = patches.view(B * N, C, Hp, Wp)
        patches_resized = F.interpolate(
            patches_flat, 
            size=self.input_resolution, 
            mode='bilinear', 
            align_corners=False
        )

        all_inputs = torch.cat([patches_resized, images], dim=0) # (B*N + B, ...)
        vae_out = self.vae(all_inputs)

        patch_reconstruction = vae_out["reconstructions"][:patch_end_idx]
        full_image_reconstruction = vae_out["reconstructions"][patch_end_idx:]

        all_box_dists = vae_out["box_distributions"]

        patch_box_dists = BoxEmbeddingDistribution(
            all_box_dists.mu_min[:patch_end_idx],
            all_box_dists.mu_max[:patch_end_idx],
            all_box_dists.beta_min[:patch_end_idx],
            all_box_dists.beta_max[:patch_end_idx]
        ) # Internal parameter shape: (B*N, D)

        full_image_box_dists = BoxEmbeddingDistribution(
            all_box_dists.mu_min[patch_end_idx:],
            all_box_dists.mu_max[patch_end_idx:],
            all_box_dists.beta_min[patch_end_idx:],
            all_box_dists.beta_max[patch_end_idx:]
        ) # Internal parameter shape: (B, D)

        patch_samples = vae_out["samples"][:patch_end_idx]
        full_image_samples = vae_out["samples"][patch_end_idx:]
        
        return {
            "images": images,
            "patch_reconstruction": patch_reconstruction,
            "full_image_reconstruction": full_image_reconstruction,
            "original_patches_resized": patches_resized,
            "patch_box_dists": patch_box_dists,
            "full_image_box_dists": full_image_box_dists,
            "patch_samples": patch_samples,
            "full_image_samples": full_image_samples,
        }

    def training_step(self, batch, batch_idx):
        output = self(batch)
        images = output["images"]

        patch_reconstruction = output["patch_reconstruction"]
        full_image_reconstruction = output["full_image_reconstruction"]

        patch_box_dists = output["patch_box_dists"] # Internal parameter shape: (B*N, D)
        full_image_box_dists = output["full_image_box_dists"] # Internal parameter shape: (B, D)

        prior_box_dists = self.prior()
        patch_prior_box_dists = prior_box_dists[0] # Internal parameter shape: (1, N_P, D)
        full_image_prior_box_dists = prior_box_dists[1] # Internal parameter shape: (1, N_FI, D)


        # ----- Pull Loss Computation -----
        patch_cluster_intersections = self.compute_cluster_intersections(patch_box_dists, patch_prior_box_dists) # Internal parameter shape: (B*N, N_P, D)
        patch_cluster_intersections_volume = bessel_volume(patch_cluster_intersections, volume_temp=0.1, log_scale=True) # (B*N, N_P)
        patch_cluster_assignment_logits = patch_cluster_intersections_volume
        patch_cluster_assignment_probs = F.softmax(patch_cluster_assignment_logits, dim = -1)
        encoded_patch_volume = bessel_volume(patch_box_dists, volume_temp=0.1, log_scale=True).unsqueeze(1) # (B*N, 1)
        
        if self.model_config["pull_method"] == "intersection":
            
            patch_cluster_volume_diff = F.relu(encoded_patch_volume - patch_cluster_intersections_volume)
            
            patch_cluster_pull_loss = torch.sum(patch_cluster_assignment_probs.detach() * patch_cluster_volume_diff, dim=1).mean()
       
        elif self.model_config["pull_method"] == "distance":
            enc_mu_min = patch_box_dists.mu_min.unsqueeze(1)
            enc_mu_max = patch_box_dists.mu_max.unsqueeze(1)
            prior_mu_min = patch_prior_box_dists.mu_min
            prior_mu_max = patch_prior_box_dists.mu_max
            
            # This pulls the 'min' corners together and 'max' corners together.
            min_dist = torch.sum((enc_mu_min - prior_mu_min)**2, dim=-1) # (B*N, N_P)
            max_dist = torch.sum((enc_mu_max - prior_mu_max)**2, dim=-1) # (B*N, N_P)
            
            enc_beta_min = patch_box_dists.beta_min.unsqueeze(1)
            enc_beta_max = patch_box_dists.beta_max.unsqueeze(1)
            prior_beta_min = patch_prior_box_dists.beta_min
            prior_beta_max = patch_prior_box_dists.beta_max

            beta_min_dist = torch.sum((enc_beta_min - prior_beta_min)**2, dim=-1)
            beta_max_dist = torch.sum((enc_beta_max - prior_beta_max)**2, dim=-1)
            
            total_geometric_dist = min_dist + max_dist# + 0.5 * (beta_min_dist + beta_max_dist)
            
            patch_cluster_pull_loss = torch.sum(patch_cluster_assignment_probs.detach() * total_geometric_dist, dim=1).mean()

        elif self.model_config["pull_method"] == "centroid":
            enc_mu_min = patch_box_dists.mu_min.unsqueeze(1)
            enc_mu_max = patch_box_dists.mu_max.unsqueeze(1)
            prior_mu_min = patch_prior_box_dists.mu_min
            prior_mu_max = patch_prior_box_dists.mu_max

            


        full_image_cluster_intersections = self.compute_cluster_intersections(full_image_box_dists, full_image_prior_box_dists)
        full_image_cluster_intersections_volume = bessel_volume(full_image_cluster_intersections, volume_temp=0.1, log_scale=True)
        full_image_cluster_assignment_logits = full_image_cluster_intersections_volume
        full_image_cluster_assignment_probs = F.softmax(full_image_cluster_assignment_logits, dim=-1)
        encoded_full_image_volume = bessel_volume(full_image_box_dists, volume_temp=0.1, log_scale=True).unsqueeze(1)

        if self.model_config["pull_method"] == "intersection":
            
            full_image_volume_diff = F.relu(encoded_full_image_volume - full_image_cluster_intersections_volume)
            full_image_cluster_pull_loss = torch.sum(full_image_cluster_assignment_probs.detach() * full_image_volume_diff, dim=1).mean()

        elif self.model_config["pull_method"] == "distance":
            enc_mu_min = full_image_box_dists.mu_min.unsqueeze(1)
            enc_mu_max = full_image_box_dists.mu_max.unsqueeze(1)
            prior_mu_min = full_image_prior_box_dists.mu_min
            prior_mu_max = full_image_prior_box_dists.mu_max

            min_dist = torch.sum((enc_mu_min - prior_mu_min)**2, dim=-1) # (B, N_FI)
            max_dist = torch.sum((enc_mu_max - prior_mu_max)**2, dim=-1) # (B, N_FI)

            enc_beta_min = full_image_box_dists.beta_min.unsqueeze(1)
            enc_beta_max = full_image_box_dists.beta_max.unsqueeze(1)
            prior_beta_min = full_image_prior_box_dists.beta_min
            prior_beta_max = full_image_prior_box_dists.beta_max

            beta_min_dist = torch.sum((enc_beta_min - prior_beta_min)**2, dim=-1)
            beta_max_dist = torch.sum((enc_beta_max - prior_beta_max)**2, dim=-1)
            
            total_geometric_dist = min_dist + max_dist# + 0.5 * (beta_min_dist + beta_max_dist)
            
            full_image_cluster_pull_loss = torch.sum(full_image_cluster_assignment_probs.detach() * total_geometric_dist, dim=1).mean()
        

        # full_image_cluster_intersections = self.compute_cluster_intersections(full_image_box_dists, full_image_prior_box_dists) # Internal parameter shape: (B, N_FI, D)
        # full_image_cluster_intersections_volume = bessel_volume(full_image_cluster_intersections, volume_temp=0.1, log_scale=True) # (B, N_FI)
        # encoded_full_image_volume = bessel_volume(full_image_box_dists, volume_temp=0.1, log_scale=True).unsqueeze(1) # (B, 1)
        # full_image_cluster_assignment_logits = full_image_cluster_intersections_volume
        # full_image_cluster_assignment_probs = F.softmax(full_image_cluster_assignment_logits, dim = -1)
        # full_image_volume_diff = F.relu(encoded_full_image_volume - full_image_cluster_intersections_volume)
        # full_image_cluster_pull_loss = torch.sum(full_image_cluster_assignment_probs.detach() * full_image_volume_diff, dim=1).mean()


        # ----- Inclusion Loss Computation -----
        B = full_image_box_dists.mu_min.shape[0]
        N_patches = patch_box_dists.mu_min.shape[0] // B
        D = patch_box_dists.mu_min.shape[-1]

        batched_patch_mu_min = patch_box_dists.mu_min.view(B, N_patches, D)
        batched_patch_mu_max = patch_box_dists.mu_max.view(B, N_patches, D)
        batched_patch_beta_min = patch_box_dists.beta_min.view(B, N_patches, D)
        batched_patch_beta_max = patch_box_dists.beta_max.view(B, N_patches, D)
        
        batched_patch_dists = BoxEmbeddingDistribution(
            batched_patch_mu_min, batched_patch_mu_max, 
            batched_patch_beta_min, batched_patch_beta_max
        )

        patch_inclusion_weights = torch.ones(B, 1, N_patches, device=images.device)
        patch_constraint_box_dist = soft_box_weighted_intersection(batched_patch_dists, patch_inclusion_weights)

        constraint_mu_min = patch_constraint_box_dist.mu_min.squeeze(1)
        constraint_mu_max = patch_constraint_box_dist.mu_max.squeeze(1)
        constraint_beta_min = patch_constraint_box_dist.beta_min.squeeze(1)
        constraint_beta_max = patch_constraint_box_dist.beta_max.squeeze(1)

        incl_beta_min = (full_image_box_dists.beta_min + constraint_beta_min) / 2.0
        incl_beta_max = (full_image_box_dists.beta_max + constraint_beta_max) / 2.0

        # Min (Smooth Max)
        arg_min_img = full_image_box_dists.mu_min / incl_beta_min
        arg_min_con = constraint_mu_min / incl_beta_min
        incl_mu_min = incl_beta_min * torch.logaddexp(arg_min_img, arg_min_con)
        
        # Max (Smooth Min)
        arg_max_img = -full_image_box_dists.mu_max / incl_beta_max
        arg_max_con = -constraint_mu_max / incl_beta_max
        incl_mu_max = -incl_beta_max * torch.logaddexp(arg_max_img, arg_max_con)
        
        # Log Volume of Image
        log_vol_img = encoded_full_image_volume.squeeze(1) # (B,)
        
        # Log Volume of Intersection
        intersection_dist = BoxEmbeddingDistribution(incl_mu_min, incl_mu_max, incl_beta_min, incl_beta_max)
        log_vol_intersection = bessel_volume(intersection_dist, volume_temp=0.1, log_scale=True) # (B,)
        
        inclusion_loss = F.relu(log_vol_img - log_vol_intersection).mean()


        # ----- Exclusion Loss -----
        flat_l1_dist = BoxEmbeddingDistribution(
            full_image_prior_box_dists.mu_min.squeeze(0),
            full_image_prior_box_dists.mu_max.squeeze(0),
            full_image_prior_box_dists.beta_min.squeeze(0),
            full_image_prior_box_dists.beta_max.squeeze(0)
        )
        
        flat_l0_dist = BoxEmbeddingDistribution(
            patch_prior_box_dists.mu_min.squeeze(0),
            patch_prior_box_dists.mu_max.squeeze(0),
            patch_prior_box_dists.beta_min.squeeze(0),
            patch_prior_box_dists.beta_max.squeeze(0)
        )

        # 2. Compute Intersection of every L1 box with every L0 box
        # Output Shape: (Num_L1, Num_L0, D)
        all_pairs_inter = pairwise_gumbel_intersection(flat_l1_dist, flat_l0_dist)
        
        # 3. Compute Volume (Linear Scale)
        # Output Shape: (Num_L1, Num_L0)
        all_pairs_log_vol = bessel_volume(all_pairs_inter, volume_temp=0.1, log_scale=True)
        all_pairs_vol = torch.exp(all_pairs_log_vol)
        adj_logits = self.prior.adjacency_logits[0] # Shape: (Num_L1, Num_L0)
        adj_weights = torch.sigmoid(adj_logits)
        exclusion_loss = ((1.0 - adj_weights.detach()) * all_pairs_vol).mean()

        # ----- Reconstruction Loss -----
        recon_loss_full_mse = F.mse_loss(full_image_reconstruction, images)
        recon_loss_full_ssim = 1 - self.ssim_loss_fn(full_image_reconstruction, images)

        patches = self.divide_image_into_patches(images) # (B, N, C, Hp, Wp)
        B, N, C, Hp, Wp = patches.shape
        patches_flat = patches.view(B * N, C, Hp, Wp)
        patches_target = F.interpolate(
            patches_flat, 
            size=self.input_resolution, 
            mode='bilinear', 
            align_corners=False
        )

        recon_loss_patch_mse = F.mse_loss(patch_reconstruction, patches_target)
        recon_loss_patch_ssim = 1 - self.ssim_loss_fn(patch_reconstruction, patches_target)


        # ----- Hebbian Loss -----
        B = full_image_cluster_assignment_probs.shape[0]
        N_patches = patch_cluster_assignment_probs.shape[0] // B
        Num_L0 = patch_cluster_assignment_probs.shape[1]
        
        patch_probs_reshaped = patch_cluster_assignment_probs.view(B, N_patches, Num_L0) # (B, N_p, N_l0)
        full_image_probs = full_image_cluster_assignment_probs # (B, N_l1)

        l0_indices = torch.argmax(patch_probs_reshaped, dim=-1) # (B, N_p)
        l1_indices = torch.argmax(full_image_probs, dim = -1) # (B,)
        l1_indices_expanded = l1_indices.unsqueeze(1).expand(-1, N_patches) # (B, N_p)

        rows = l1_indices_expanded.reshape(-1)
        cols = l0_indices.reshape(-1)

        target_adj = torch.zeros_like(self.prior.adjacency_logits[0], device = self.device) # (N_l1, N_l0)
        # target_adj[rows, cols] = 1.0
        target_adj.index_put_((rows, cols), torch.tensor(1.0, device=self.device))
        N_l1, N_l0 = target_adj.shape

        active_l1_mask = torch.zeros(N_l1, dtype=torch.bool, device=self.device)
        active_l1_mask[l1_indices] = True
        
        # Current Logits: (Num_L1, Num_L0)
        adj_logits = self.prior.adjacency_logits[0]
        
        # Select only the active rows
        active_logits = adj_logits[active_l1_mask]      # (Num_Active, Num_L0)
        active_targets = target_adj[active_l1_mask]     # (Num_Active, Num_L0)
        
        if active_logits.shape[0] > 0:
            hebbian_loss = F.binary_cross_entropy_with_logits(active_logits, active_targets)
        else:
            hebbian_loss = 0.0


        # ----- Regularization Losses -----
        
        # --- Uniform Prior Loss ---
        loss_uniform_patch = self.compute_marginal_entropy_loss(patch_cluster_assignment_probs)
        loss_uniform_full = self.compute_marginal_entropy_loss(full_image_cluster_assignment_probs)

        # --- Disjoint Prior Loss ---
        loss_disjoint_full = self.compute_pairwise_disjoint_loss(prior_box_dists[1])
        loss_disjoint_patch = self.compute_pairwise_disjoint_loss(prior_box_dists[0])

        # --- Entropy Loss ---
        adj_logits = self.prior.adjacency_logits[0]
        adj_probs = torch.sigmoid(adj_logits)
        
        # Standard Entropy: -p*log(p) - (1-p)*log(1-p)
        entropy = -adj_probs * torch.log(adj_probs + 1e-10) - (1 - adj_probs) * torch.log(1 - adj_probs + 1e-10)
        entropy_loss = entropy.mean()

        # --- l0 and l1 assignment entropy ---
        avg_l0_usage = patch_cluster_assignment_probs.mean(dim = 0)
        l0_entropy = -torch.sum(avg_l0_usage * torch.log(avg_l0_usage + 1e-10))
        l0_entropy_loss = -l0_entropy

        avg_l1_usage = full_image_cluster_assignment_probs.mean(dim = 0)
        l1_entropy = -torch.sum(avg_l1_usage * torch.log(avg_l1_usage + 1e-10))
        l1_entropy_loss = -l1_entropy

        # --- Box Volume Penalty ---
        l0_vol = bessel_volume(flat_l0_dist, log_scale=False)
        l0_vol_penalty = l0_vol.mean()

        l1_vol = bessel_volume(flat_l1_dist, log_scale=False)
        l1_vol_penalty = l1_vol.mean()

        if "hebbian_start_epoch" in self.model_config:
            if self.current_epoch >= self.model_config["hebbian_start_epoch"]:
                enable_hebbian = 1.0
            else:
                enable_hebbian = 0.0
        else:
            enable_hebbian = 1.0

        if "full_image_start_epoch" in self.model_config:
            if self.current_epoch >= self.model_config["full_image_start_epoch"]:
                enable_full_image = 1.0
            else:
                enable_full_image = 0.0
        else:
            enable_full_image = 1.0


        total_loss = (
            self.loss_weights["patch_mse_loss"] * recon_loss_patch_mse +
            self.loss_weights["patch_ssim_loss"] * recon_loss_patch_ssim +
            self.loss_weights["full_image_mse_loss"] * recon_loss_full_mse * enable_full_image +
            self.loss_weights["full_image_ssim_loss"] * recon_loss_full_ssim * enable_full_image +
            self.loss_weights["patch_pull_loss"] * patch_cluster_pull_loss +
            self.loss_weights["full_image_pull_loss"] * full_image_cluster_pull_loss * enable_full_image +
            self.loss_weights["inclusion_loss"] * inclusion_loss * enable_full_image +
            self.loss_weights["exclusion_loss"] * exclusion_loss * enable_full_image +
            self.loss_weights["loss_uniform_patch"] * loss_uniform_patch + 
            self.loss_weights["loss_uniform_full"] * loss_uniform_full * enable_full_image + 
            self.loss_weights["loss_disjoint_full"] * loss_disjoint_full +
            self.loss_weights["loss_disjoint_patch"] * loss_disjoint_patch + 
            self.loss_weights["entropy_loss"] * entropy_loss +
            self.loss_weights["hebbian_loss"] * hebbian_loss * enable_hebbian +
            self.loss_weights["l0_entropy_loss"] * l0_entropy_loss +
            self.loss_weights["l1_entropy_loss"] * l1_entropy_loss +
            self.loss_weights["l0_vol_penalty"] * l0_vol_penalty + 
            self.loss_weights["l1_vol_penalty"] * l1_vol_penalty
        )

        loss_dict = {
            "total_loss": total_loss,
            "patch_pull_loss": patch_cluster_pull_loss,
            "full_image_pull_loss": full_image_cluster_pull_loss,
            "inclusion_loss": inclusion_loss,
            "exclusion_loss": exclusion_loss,
            "patch_mse_loss": recon_loss_patch_mse,
            "patch_ssim_loss": recon_loss_patch_ssim,
            "full_image_mse_loss": recon_loss_full_mse,
            "full_image_ssim_loss": recon_loss_full_ssim,
            "loss_uniform_patch": loss_uniform_patch,
            "loss_uniform_full": loss_uniform_full,
            "loss_disjoint_patch": loss_disjoint_patch,
            "loss_disjoint_full": loss_disjoint_full,
            "entropy_loss": entropy_loss,
            "hebbian_loss": hebbian_loss,
            "l0_entropy_loss": l0_entropy_loss,
            "l1_entropy_loss": l1_entropy_loss,
            "l0_vol_penalty": l0_vol_penalty,
            "l1_vol_penalty": l1_vol_penalty
        }
        self.log_dict(loss_dict, prog_bar=True, on_epoch=True)

        if self.viz_datapoint is None:
            N = self.grid_size[0] * self.grid_size[1]
            
            self.viz_datapoint = {
                "original_image": images[0].detach().cpu(),
                "recon_image": output["full_image_reconstruction"][0].detach().cpu(),
                "input_patches": output["original_patches_resized"][:N].detach().cpu(),
                "recon_patches": output["patch_reconstruction"][:N].detach().cpu()
            }

        return total_loss


    def compute_cluster_intersections(self, encoded_box_dists, prior_box_dists):
        # Encoded box dists have internal parameter shape (B, D)
        # Prior box dists have internal parameter shape (1, N, D)

        # Reshape encoded box dists internal parameters to (B, 1, D)
        encoded_box_dist_mu_min = encoded_box_dists.mu_min.unsqueeze(1) 
        encoded_box_dist_mu_max = encoded_box_dists.mu_max.unsqueeze(1)
        encoded_box_dist_beta_min = encoded_box_dists.beta_min.unsqueeze(1)
        encoded_box_dist_beta_max = encoded_box_dists.beta_max.unsqueeze(1)

        # Extract prior box dists internal parameters. Shape: (1, N, D)
        prior_box_dist_mu_min = prior_box_dists.mu_min
        prior_box_dist_mu_max = prior_box_dists.mu_max
        prior_box_dist_beta_min = prior_box_dists.beta_min
        prior_box_dist_beta_max = prior_box_dists.beta_max

        new_beta_min = (encoded_box_dist_beta_min + prior_box_dist_beta_min) / 2.0
        new_beta_max = (encoded_box_dist_beta_max + prior_box_dist_beta_max) / 2.0

        encoded_arg_min = encoded_box_dist_mu_min / new_beta_min
        prior_arg_min = prior_box_dist_mu_min / new_beta_min
        new_mu_min = new_beta_min * torch.logaddexp(encoded_arg_min, prior_arg_min)

        encoded_arg_max = -encoded_box_dist_mu_max / new_beta_max
        prior_arg_max = -prior_box_dist_mu_max / new_beta_max
        new_mu_max = -new_beta_max * torch.logaddexp(encoded_arg_max, prior_arg_max)

        return BoxEmbeddingDistribution(new_mu_min, new_mu_max, new_beta_min, new_beta_max)

    def compute_marginal_entropy_loss(self, assignment_probs):
        avg_probs = assignment_probs.mean(dim=0)
        
        num_clusters = avg_probs.shape[0]
        target_probs = torch.ones_like(avg_probs) / num_clusters
        
        kl_div = torch.sum(avg_probs * torch.log((avg_probs + 1e-10) / (target_probs + 1e-10)))
        
        return kl_div

    def compute_pairwise_disjoint_loss(self, prior_boxes, volume_temp=0.1):
        """
        Penalizes overlap between different clusters in the Prior.
        Args:
            prior_boxes: BoxEmbeddingDistribution of the Prior Clusters (Level 1)
        """
        # 1. Compute Pairwise Intersection of Prior Boxes with THEMSELVES
        # Output: (N_Clusters, N_Clusters, Dim)
        # We reuse your intersection logic but applied to (N, 1) and (1, N)
        n_clusters = prior_boxes.mu_min.shape[1] # Internal shape is (1, N, D)
        
        # Flatten internal params to (N, D) for the function
        flat_mu_min = prior_boxes.mu_min.squeeze(0)
        flat_mu_max = prior_boxes.mu_max.squeeze(0)
        flat_beta_min = prior_boxes.beta_min.squeeze(0)
        flat_beta_max = prior_boxes.beta_max.squeeze(0)
        
        flat_dist = BoxEmbeddingDistribution(flat_mu_min, flat_mu_max, flat_beta_min, flat_beta_max)
        
        # Compute Intersection Table (N, N)
        intersection_dist = pairwise_gumbel_intersection(flat_dist, flat_dist)
        
        # 2. Compute Volume of Intersections
        # Shape: (N, N)
        log_volumes = bessel_volume(intersection_dist, volume_temp=volume_temp, log_scale=True)
        volumes = torch.exp(log_volumes)
        
        # 3. Mask out the diagonal (Self-intersection is allowed/expected)
        mask = torch.eye(n_clusters, device=volumes.device).bool()
        volumes_masked = volumes.masked_fill(mask, 0.0)
        
        # 4. Loss = Sum of all off-diagonal volumes
        # We want this to be 0 (Disjoint)
        disjoint_loss = volumes_masked.sum() / (n_clusters * (n_clusters - 1))

        centers = (flat_mu_min + flat_mu_max) / 2.0

        r = torch.sum(centers**2, dim=1).view(-1, 1)
        dist_sq = r - 2.0 * torch.matmul(centers, centers.t()) + r.t()
        dist_sq = F.relu(dist_sq) # Clamp negative noise
        dist = torch.sqrt(dist_sq + 1e-8)

        # 3. Hinge Loss: Penalize if distance < Threshold
        # We want every pair to be at least 'margin' apart
        margin = 0.5
        
        # Mask diagonal (dist to self is 0, which is fine)
        n = centers.shape[0]
        mask_ = 1.0 - torch.eye(n, device=centers.device)
        
        # Loss = sum(ReLU(margin - dist)) for all off-diagonal pairs
        repulsion_loss = (mask_ * F.relu(margin - dist)).sum() / (n * (n-1))
        
        return disjoint_loss + repulsion_loss

    def divide_image_into_patches(self, images):
        B, C, H, W = images.shape
        grid_h, grid_w = self.grid_size
        patch_h = H // grid_h
        patch_w = W // grid_w

        if H % grid_h != 0 or W % grid_w != 0:
            raise ValueError("Image dimensions must be divisible by grid size.")

        patches = images.view(
            B, C, grid_h, patch_h, grid_w, patch_w
        ) 
        patches = patches.permute(0, 2, 4, 1, 3, 5)  
        patches = patches.reshape(B, -1, C, patch_h, patch_w)
        
        return patches

    def configure_optimizers(self):

        if self.config["trainer"]["optimizer"]["type"] == "Adam":
            # return torch.optim.Adam(
            #     self.parameters(),
            #     **self.config["trainer"]["optimizer"]["config"],
            # )
            optimizer = torch.optim.Adam([
                {'params': self.vae.parameters(), 'lr': self.config["trainer"]["optimizer"]["config"]["vae_lr"]},
                {'params': self.prior.parameters(), 'lr': self.config["trainer"]["optimizer"]["config"]["prior_lr"]}
            ])
            return optimizer
        else:
            raise ValueError(
                f"Optimizer type {self.config['trainer']['optimizer']['type']} not implemented."
            )

    def on_train_epoch_end(self):
        if self.viz_datapoint is not None:
            fig = self.create_reconstruction_figure(
                self.viz_datapoint, 
                self.grid_size, 
                self.current_epoch
            )
            self.logger.experiment.log({"reconstruction_visualization matrix": fig})
            
            self.viz_datapoint = None
            plt.close(fig)
        
        with torch.no_grad():
            prior_dists = self.prior()
            level0_dist = prior_dists[0]
            level1_dist = prior_dists[1]

            adj_logits = self.prior.adjacency_logits[0]
            adj_weights = torch.sigmoid(adj_logits).detach().cpu().numpy()
            
            fig_adj = plt.figure(figsize=(10, 8))
            plt.imshow(adj_weights, aspect='auto', cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label="Connection Probability")
            plt.xlabel(f"Level 0 Boxes (Patches) [Count: {adj_weights.shape[1]}]")
            plt.ylabel(f"Level 1 Boxes (Images) [Count: {adj_weights.shape[0]}]")
            plt.title(f"Learned Adjacency (Epoch {self.current_epoch})")
            plt.tight_layout()
            self.logger.experiment.log({"adjacency matrix": fig_adj})


            recon_l0 = self.decode_prior_boxes(level0_dist)
            recon_l1 = self.decode_prior_boxes(level1_dist)
            fig_l0 = self.create_grid_figure(recon_l0, f"Level 0 (Patches) Prior - Epoch {self.current_epoch}")
            fig_l1 = self.create_grid_figure(recon_l1, f"Level 1 (Full Images) Prior - Epoch {self.current_epoch}")
            self.logger.experiment.log({"Level 0 Prior Reconstructions": fig_l0})
            self.logger.experiment.log({"Level 1 Prior Reconstructions": fig_l1})

            plt.close(fig_adj)
            plt.close(fig_l0)
            plt.close(fig_l1)
            

    def decode_prior_boxes(self, box_dist):
        """
        Helper to decode the canonical representation (mean) of a box distribution.
        """
        mu_min = box_dist.mu_min.squeeze(0)
        mu_max = box_dist.mu_max.squeeze(0)
        
        z = torch.cat([mu_min, mu_max], dim=-1)
        
        return self.vae.decode(z)

    @staticmethod
    def create_grid_figure(images, title):
        """Helper to create a simple grid figure"""

        max_imgs = 64
        if images.shape[0] > max_imgs:
            images = images[:max_imgs]
            
        grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(np.clip(grid_np, 0, 1))
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        return fig

    @staticmethod
    def create_reconstruction_figure(data, grid_size, epoch):
        """
        Creates a matplotlib figure comparing original and reconstructed images/patches.
        """
        original_img = data["original_image"].permute(1, 2, 0).numpy()
        recon_img = data["recon_image"].permute(1, 2, 0).numpy()
        
        input_patches = data["input_patches"]
        recon_patches = data["recon_patches"]
        
        fig = plt.figure(figsize=(12, 8))
        plt.suptitle(f"Epoch {epoch} Reconstruction", fontsize=16)

        # 1. Full Image Original
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(np.clip(original_img, 0, 1))
        ax1.set_title("Original Image")
        ax1.axis("off")

        # 2. Full Image Reconstruction
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(np.clip(recon_img, 0, 1))
        ax2.set_title("Recon Image")
        ax2.axis("off")

        # 3. Input Patches Grid
        grid_in = torchvision.utils.make_grid(input_patches, nrow=grid_size[1], padding=2)
        grid_in = grid_in.permute(1, 2, 0).numpy()
        
        ax3 = plt.subplot(2, 4, 3)
        ax3.imshow(np.clip(grid_in, 0, 1))
        ax3.set_title(f"Input Patches\n(Resized to {input_patches.shape[-1]}x{input_patches.shape[-1]})")
        ax3.axis("off")

        # 4. Recon Patches Grid
        grid_recon = torchvision.utils.make_grid(recon_patches, nrow=grid_size[1], padding=2)
        grid_recon = grid_recon.permute(1, 2, 0).numpy()

        ax4 = plt.subplot(2, 4, 4)
        ax4.imshow(np.clip(grid_recon, 0, 1))
        ax4.set_title("Recon Patches")
        ax4.axis("off")

        plt.tight_layout()
        return fig

    def on_after_backward(self):
        if self.trainer.global_step % 100 == 0:  # Don't spam logs
            for i, param in enumerate(self.prior.adjacency_logits):
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"Step {self.trainer.global_step}: Adjacency Level {i} Grad Norm: {grad_norm}")
                    print(self.prior.adjacency_logits[0].sigmoid().mean())
                    
                else:
                    print(f"Step {self.trainer.global_step}: Adjacency Level {i} Grad is None!")