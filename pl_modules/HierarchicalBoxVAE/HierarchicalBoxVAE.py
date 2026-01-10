import random

import torch
import torchvision
from torch import log_, nn
from torch.nn import functional as F
import lightning as L
import numpy as np
from matplotlib import pyplot as plt

from pytorch_msssim import SSIM

from .BoxEmbedVAE import BoxEmbedVAE
from .HierarchicalBoxPrior import HierarchicalBoxPrior
from .box_utils import BoxDistribution, soft_box_weighted_intersection, pairwise_gumbel_intersection, bessel_volume

class HierarchicalBoxVAE(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config["model"]["config"]
        self.vae_config = self.model_config["vae_config"]
        self.prior_config = self.model_config["prior_config"]

        self.embed_dim = self.model_config["embed_dim"]
        self.grid_size = self.model_config["grid_size"]
        self.input_resolution = self.model_config["input_resolution"]
        self.crop_padding_range = self.model_config["crop_padding_range"]
        # self.pull_method = self.model_config["pull_method"]

        self.beta_scale = self.model_config["beta_scale"]
        self.beta_pre_init = self.model_config["beta_pre_init"]
        self.beta_activation = self.model_config["beta_activation"]

        self.vae = BoxEmbedVAE(
            vae_config=self.vae_config,
            embed_dim=self.embed_dim,
            input_resolution=self.input_resolution,
            beta_scale = self.beta_scale,
            beta_pre_init = self.beta_pre_init,
        )

        self.prior = HierarchicalBoxPrior(
            prior_config = self.prior_config,
            embed_dim = self.embed_dim,
            beta_scale = self.beta_scale,
            beta_pre_init = self.beta_pre_init,
            beta_activation = self.beta_activation
        )

        self.loss_weights = self.model_config["loss_weights"]
        self.ssim_loss_fn = SSIM(
            win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3
        )

        self.viz_datapoint = None

    def forward(self, x):
        images = x["images"]
        masks = x["object_masks"]

        images = self.crop_to_content(images, masks)

        patches = self.divide_image_into_patches(images) # (B, N, C, H_p, W_p)
        B, N, C, Hp, Wp = patches.shape
        num_total_patches = B * N

        patches_flat = patches.view(B * N, C, Hp, Wp)
        patches_resized = F.interpolate(
            patches_flat,
            size=self.input_resolution, 
            mode='bilinear', 
            align_corners=False
        )

        all_inputs = torch.cat([patches_resized, images], dim=0) # (B*N + B, ...)
        vae_out = self.vae(all_inputs)

        patch_reconstructions = vae_out["reconstructions"][:num_total_patches]
        full_image_reconstructions = vae_out["reconstructions"][num_total_patches:]

        all_box_dists = vae_out["box_distributions"]

        patch_box_dists = BoxDistribution(
            all_box_dists.mu_min[:num_total_patches],
            all_box_dists.mu_max[:num_total_patches],
            all_box_dists.beta_min[:num_total_patches],
            all_box_dists.beta_max[:num_total_patches]
        ) # Internal parameter shape: (B*N, D)

        full_image_box_dists = BoxDistribution(
            all_box_dists.mu_min[num_total_patches:],
            all_box_dists.mu_max[num_total_patches:],
            all_box_dists.beta_min[num_total_patches:],
            all_box_dists.beta_max[num_total_patches:]
        ) # Internal parameter shape: (B, D)

        patch_samples = vae_out["samples"][:num_total_patches]
        full_image_samples = vae_out["samples"][num_total_patches:]

        return {
            "images": images,
            "patches": patches_resized,
            "patch_reconstructions": patch_reconstructions,
            "full_image_reconstructions": full_image_reconstructions,
            "patch_box_dists": patch_box_dists,
            "full_image_box_dists": full_image_box_dists,
            "patch_samples": patch_samples,
            "full_image_samples": full_image_samples,
        }

    def training_step(self, batch, batch_idx):
        output = self(batch)
        images = output["images"]
        patches = output["patches"]

        patch_reconstructions = output["patch_reconstructions"]
        full_image_reconstructions = output["full_image_reconstructions"]

        patch_box_dists = output["patch_box_dists"] # Internal parameter shape: (B*N, D)
        full_image_box_dists = output["full_image_box_dists"] # Internal parameter shape: (B, D)

        patch_samples = output["patch_samples"]
        full_image_samples = output["full_image_samples"]

        prior_box_dists = self.prior()
        patch_prior_box_dists = prior_box_dists[0] # Internal parameter shape: (1, N_L0, D)
        patch_prior_box_dists_flat = BoxDistribution(
            patch_prior_box_dists.mu_min.squeeze(0),
            patch_prior_box_dists.mu_max.squeeze(0),
            patch_prior_box_dists.beta_min.squeeze(0),
            patch_prior_box_dists.beta_max.squeeze(0)
        ) # Internal parameter shape: (N_L0, D)
        full_image_prior_box_dists = prior_box_dists[1] # Internal parameter shape: (1, N_L1, D)
        full_image_prior_box_dists_flat = BoxDistribution(
            full_image_prior_box_dists.mu_min.squeeze(0),
            full_image_prior_box_dists.mu_max.squeeze(0),
            full_image_prior_box_dists.beta_min.squeeze(0),
            full_image_prior_box_dists.beta_max.squeeze(0)
        ) # Internal parameter shape: (N_L1, D)


        # ----- LOSS COMPUTATIONS ----
        # --- Pull Loss ---
        patch_pull_loss, patch_prior_assignment_probs = self.compute_pull_loss(
            computed_box_dists = patch_box_dists,
            computed_box_samples = patch_samples,
            prior_box_dists = patch_prior_box_dists_flat
        )
        full_image_pull_loss, full_image_prior_assignment_probs = self.compute_pull_loss(
            computed_box_dists = full_image_box_dists,
            computed_box_samples = full_image_samples,
            prior_box_dists = full_image_prior_box_dists_flat
        )

        # --- Reconstruction Loss ---
        recon_loss_full_mse, recon_loss_full_ssim = self.compute_reconstruction_loss(images, full_image_reconstructions)
        recon_loss_patch_mse, recon_loss_patch_ssim = self.compute_reconstruction_loss(patches, patch_reconstructions)

        # --- Entropy Loss ---
        avg_l0_usage = patch_prior_assignment_probs.mean(dim = 0)
        l0_entropy = -torch.sum(avg_l0_usage * torch.log(avg_l0_usage + 1e-10))
        l0_entropy_loss = -l0_entropy

        avg_l1_usage = full_image_prior_assignment_probs.mean(dim = 0)
        l1_entropy = -torch.sum(avg_l1_usage * torch.log(avg_l1_usage + 1e-10))
        l1_entropy_loss = -l1_entropy

        # --- Box Volume Penalty ---
        l0_vol = bessel_volume(patch_prior_box_dists_flat, log_scale=False)
        l0_vol_penalty = l0_vol.mean()

        l1_vol = bessel_volume(full_image_prior_box_dists_flat, log_scale=False)
        l1_vol_penalty = l1_vol.mean()

        # --- Disjoint Loss ---
        l0_disjoint_loss = self.compute_disjoint_loss(patch_prior_box_dists_flat)
        l1_disjoint_loss = self.compute_disjoint_loss(full_image_prior_box_dists_flat)

        patch_disjoint_loss = self.compute_disjoint_loss(patch_box_dists)
        full_image_disjoint_loss = self.compute_disjoint_loss(full_image_box_dists)


        # ----- LOSS AGGREGATION ----

        if "pull_loss_start_epoch" in self.model_config:
            if self.current_epoch >= self.model_config["pull_loss_start_epoch"]:
                enable_pull_loss = 1.0
                if "pull_loss_warmup_epochs" in self.model_config:
                    pull_loss_warmup_factor = min(1.0, (self.current_epoch - self.model_config["pull_loss_start_epoch"])/self.model_config["pull_loss_warmup_epochs"])
                else:
                    pull_loss_warmup_factor = 1.0
            else:
                enable_pull_loss = 0.0
                pull_loss_warmup_factor = 1.0
        else:
            enable_pull_loss = 1.0
            pull_loss_warmup_factor = 1.0


        total_loss = (
            self.loss_weights["patch_mse_loss"] * recon_loss_patch_mse +
            self.loss_weights["patch_ssim_loss"] * recon_loss_patch_ssim +
            self.loss_weights["full_image_mse_loss"] * recon_loss_full_mse +
            self.loss_weights["full_image_ssim_loss"] * recon_loss_full_ssim +
            self.loss_weights["patch_pull_loss"] * patch_pull_loss * enable_pull_loss * pull_loss_warmup_factor +
            self.loss_weights["full_image_pull_loss"] * full_image_pull_loss * enable_pull_loss * pull_loss_warmup_factor + 
            self.loss_weights["l0_entropy_loss"] * l0_entropy_loss + 
            self.loss_weights["l1_entropy_loss"] * l1_entropy_loss + 
            self.loss_weights["l0_vol_penalty"] * l0_vol_penalty +
            self.loss_weights["l1_vol_penalty"] * l1_vol_penalty + 
            self.loss_weights["l0_disjoint_loss"] * l0_disjoint_loss +
            self.loss_weights["l1_disjoint_loss"] * l1_disjoint_loss + 
            self.loss_weights["patch_disjoint_loss"] * patch_disjoint_loss + 
            self.loss_weights["full_image_disjoint_loss"] * full_image_disjoint_loss
        )

        loss_dict = {
            "total_loss": total_loss,
            "patch_mse_loss": recon_loss_patch_mse,
            "patch_ssim_loss": recon_loss_patch_ssim,
            "full_image_mse_loss": recon_loss_full_mse,
            "full_image_ssim_loss": recon_loss_full_ssim,
            "patch_pull_loss": patch_pull_loss,
            "full_image_pull_loss": full_image_pull_loss,
            "l0_entropy_loss": l0_entropy_loss,
            "l1_entropy_loss": l1_entropy_loss,
            "l0_vol_penalty": l0_vol_penalty,
            "l1_vol_penalty": l1_vol_penalty,
            "l0_disjoint_loss": l0_disjoint_loss,
            "l1_disjoint_loss": l1_disjoint_loss,
            "patch_disjoint_loss": patch_disjoint_loss,
            "full_image_disjoint_loss": full_image_disjoint_loss

        }
        self.log_dict(loss_dict, prog_bar=True, on_epoch=True)

        if self.viz_datapoint is None:
            N = self.grid_size[0] * self.grid_size[1]
            self.viz_datapoint = {
                "original_image": images[0].detach().cpu(),
                "recon_image": output["full_image_reconstructions"][0].detach().cpu(),
                "input_patches": patches[:N].detach().cpu(),
                "recon_patches": output["patch_reconstructions"][:N].detach().cpu()
            }

        usage_counts = patch_prior_assignment_probs.sum(dim=0) # Shape: (M,)
        
        # Log usage as a histogram or just the standard deviation of usage (0 = perfect balance)
        usage_std = usage_counts.std()
        self.log("metrics/prior_usage_std", usage_std)
        
        # Log the count of "Dead Boxes" (boxes used < 1% of the time)
        threshold = patch_prior_assignment_probs.shape[0] * 0.01
        dead_boxes = (usage_counts < threshold).sum().float()
        self.log("metrics/dead_boxes", dead_boxes)

        # 2. LOG SAMPLE DISPERSION (Is the Encoder collapsing?)
        # patch_samples shape: (B*N, D*2)
        # Calculate how spread out the samples are in the latent space
        sample_spread = patch_samples.std(dim=0).mean()
        self.log("metrics/sample_spread", sample_spread)
        
        # 3. LOG DISJOINT MAGNITUDE vs PULL MAGNITUDE (Diagnostic)
        # This gives us a proxy for which loss is "winning"
        self.log("debug/l0_disjoint_val", l0_disjoint_loss.detach(), on_step=True)
        self.log("debug/patch_pull_val", patch_pull_loss.detach(), on_step=True)

        return total_loss

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

            y_min_pad = max(0, y_min - random.randint(self.crop_padding_range[0], self.crop_padding_range[1]))
            y_max_pad = min(H - 1, y_max + random.randint(self.crop_padding_range[0], self.crop_padding_range[1]))
            x_min_pad = max(0, x_min - random.randint(self.crop_padding_range[0], self.crop_padding_range[1]))
            x_max_pad = min(W - 1, x_max + random.randint(self.crop_padding_range[0], self.crop_padding_range[1]))
            
            crop = img[:, y_min_pad : y_max_pad + 1, x_min_pad : x_max_pad + 1]
            
            crop_resized = F.interpolate(
                crop.unsqueeze(0), 
                size=self.input_resolution, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            cropped_batch.append(crop_resized)
            
        return torch.stack(cropped_batch)

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

    def compute_pull_loss(self, computed_box_dists, computed_box_samples, prior_box_dists):
        """
        computed_box_dists have internal parameter shape of (N, D)
        computed_box_samples have shape (N, D*2)
        prior_box_dists have internal parameter shape of (M, D)
        """
        assignment_temp = self.model_config.get("pull_assignment_temperature", 1.0)
        assignment_activation = self.model_config.get("pull_assignment_activation", "softmax")
        assignment_hard = self.model_config.get("pull_assignment_hard", False) 

        intersection_box_dists = pairwise_gumbel_intersection(
            computed_box_dists,
            prior_box_dists
        ) # Internal parameter shape: (N, M, D)
        intersection_volume = bessel_volume(intersection_box_dists, volume_temp = 0.1, log_scale = True) # (N, M)
        if assignment_activation == "softmax":
            prior_assignment_probs = F.softmax(intersection_volume / assignment_temp, dim = -1).detach() # (N, M)
        elif assignment_activation == "gumbel_softmax":
            prior_assignment_probs = F.gumbel_softmax(intersection_volume, tau=assignment_temp, hard=assignment_hard, dim = -1) # (N, M)


        samples_min = computed_box_samples[..., :self.embed_dim].unsqueeze(1) # (N, 1, D)
        samples_max = computed_box_samples[..., self.embed_dim:].unsqueeze(1) # (N, 1, D)

        prior_box_dists_expanded = BoxDistribution(
            prior_box_dists.mu_min.unsqueeze(0),
            prior_box_dists.mu_max.unsqueeze(0),
            prior_box_dists.beta_min.unsqueeze(0),
            prior_box_dists.beta_max.unsqueeze(0)
        ) # Internal parameter shape: (1, M, D)

        if self.model_config["pull_loss_method"] == "box_likelihood":
            log_probs = self.compute_box_log_prob(
                samples_min,
                samples_max,
                prior_box_dists_expanded
            ) # (N, M, D)
            log_likelihoods = log_probs.sum(dim=-1) # (N, M)
            per_sample_loss = -torch.sum(prior_assignment_probs * log_likelihoods, dim=1) # (N,)

        elif self.model_config["pull_loss_method"] == "containment":
            violation_min = F.relu(prior_box_dists_expanded.mu_min - samples_min) # (N, M, D)
            violation_max = F.relu(samples_max - prior_box_dists_expanded.mu_max) # (N, M, D)
            
            geometric_distance = (violation_min + violation_max).sum(dim=-1) # (N, M)
            per_sample_loss = torch.sum(prior_assignment_probs * geometric_distance, dim=1) # (N,)

        elif self.model_config["pull_loss_method"] == "corner_distance":
            min_corner_dist = torch.sum((prior_box_dists_expanded.mu_min - samples_min)**2, dim = -1) # (N, M)
            max_corner_dist = torch.sum((prior_box_dists_expanded.mu_max - samples_max)**2, dim = -1) # (N, M)

            total_corner_dist = min_corner_dist + max_corner_dist # (N, M)
            per_sample_loss = torch.sum(total_corner_dist, dim = 1) # (N,)
        
        else:
            raise ValueError(
                f"Pull Loss method {self.model_config["pull_loss_method"]} not implemented."
            )

        pull_loss = per_sample_loss.mean()

        return pull_loss, prior_assignment_probs

    def compute_box_log_prob(self, sample_min: torch.Tensor, sample_max: torch.Tensor, box_dists: BoxDistribution, eps: float = 1e-10) -> torch.Tensor:
        """
        Computes the log-probability of a sampled box under a BoxDistribution.
        
        Args:
            sample_min: Shape (N, M, D) - Broadcasted samples
            sample_max: Shape (N, M, D) - Broadcasted samples
            box_dists: BoxDistribution object with parameters of shape (N, M, D)
            
        Returns:
            total_log_prob: Shape (N, M, D) - The log prob per dimension
        """

        # --- Min Coordinate (Standard Gumbel) ---
        # Log PDF: -log(beta) - z - exp(-z), where z = (x - mu) / beta
        z_min = (sample_min - box_dists.mu_min) / (box_dists.beta_min + eps)
        log_pdf_min = -torch.log(box_dists.beta_min + eps) - z_min - torch.exp(-z_min)
        
        # --- Max Coordinate (Reversed Gumbel) ---
        # Log PDF: -log(beta) + z - exp(z), where z = (x - mu) / beta
        z_max = (sample_max - box_dists.mu_max) / (box_dists.beta_max + eps)
        log_pdf_max = -torch.log(box_dists.beta_max + eps) + z_max - torch.exp(z_max)
        
        # Total Log Probability is the sum of independent log probs of corners
        total_log_prob = log_pdf_min + log_pdf_max
        
        return total_log_prob

    def compute_reconstruction_loss(self, original_images, reconstructed_images):

        recon_mse_loss = F.mse_loss(reconstructed_images, original_images)
        recon_ssim_loss = 1 - self.ssim_loss_fn(reconstructed_images, original_images)

        return recon_mse_loss, recon_ssim_loss

    def compute_disjoint_loss(self, box_dists):

        total_disjoint_loss = torch.tensor(0.0, device=self.device)
        num_box_dists = box_dists.mu_min.shape[0]

        if "intersection_volume" in self.model_config["disjoint_loss_method"]:
            intersection_box_dists = pairwise_gumbel_intersection(
                box_dists,
                box_dists
            ) # Internal parameter shape: (N, N, D)
            log_intersection_volumes = bessel_volume(intersection_box_dists, volume_temp = 0.1, log_scale = True) # (N, N)
            intersection_volumes = torch.exp(log_intersection_volumes) # (N, N)

            mask = torch.eye(num_box_dists, device=intersection_volumes.device).bool()
            volumes_masked = intersection_volumes.masked_fill(mask, 0.0)
            total_disjoint_loss += (volumes_masked.sum() / (num_box_dists * (num_box_dists - 1)))*100

        elif "centroid_distance" in self.model_config["disjoint_loss_method"]:
            # box_dist_centroids = (box_dists.mu_min + box_dists.mu_max)/2 # (N, D)
            # box_dist_centroids_A = box_dist_centroids.unsqueeze(0) # (1, N, D)
            # box_dist_centroids_B = box_dist_centroids.unsqueeze(1) # (N, 1, D)

            # centroid_distances = torch.sum((box_dist_centroids_A - box_dist_centroids_B)**2, dim = -1) # (N, N)
            # mask = torch.eye(num_box_dists, device=centroid_distances.device).bool()
            # centroid_distances_masked = centroid_distances.masked_fill(mask, 0.0)
            # total_disjoint_loss += -(centroid_distances_masked.sum() / (num_box_dists * (num_box_dists - 1)))

            box_dist_centroids = (box_dists.mu_min + box_dists.mu_max) / 2
            dists = torch.cdist(box_dist_centroids, box_dist_centroids, p=2) 
            
            margin = 0.3
            pairwise_repulsion = F.relu(margin - dists)
            
            mask = torch.eye(num_box_dists, device=pairwise_repulsion.device).bool()
            pairwise_repulsion = pairwise_repulsion.masked_fill(mask, 0.0)
            
            total_disjoint_loss += pairwise_repulsion.sum() / (num_box_dists * (num_box_dists - 1))

        return total_disjoint_loss

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