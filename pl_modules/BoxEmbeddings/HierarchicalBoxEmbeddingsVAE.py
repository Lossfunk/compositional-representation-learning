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
        
        self.vae = BoxEmbedVAE(
            latent_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            input_resolution=self.input_resolution
        )

        self.prior = HierarchicalBoxEmbeddingsPrior(
            boxes_per_level=self.model_config["prior_config"]["boxes_per_level"],
            embed_dim=self.embed_dim
        )

        self.loss_weights = self.model_config["loss_weights"]
        self.ssim_loss_fn = SSIM(
            win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3
        )

        self.viz_datapoint = None

    def forward(self, x):
        images = x["images"]
        batch_size = images.shape[0]

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

        # patch_box_dists = vae_out["box_distributions"][:patch_end_idx]
        # full_image_box_dists = vae_out["box_distributions"][patch_end_idx:]

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
            "patch_reconstruction": patch_reconstruction,
            "full_image_reconstruction": full_image_reconstruction,
            "original_patches_resized": patches_resized,
            "patch_box_dists": patch_box_dists,
            "full_image_box_dists": full_image_box_dists,
            "patch_samples": patch_samples,
            "full_image_samples": full_image_samples
        }

    def training_step(self, batch, batch_idx):
        output = self(batch)
        images = batch["images"]

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
        encoded_patch_volume = bessel_volume(patch_box_dists, volume_temp=0.1, log_scale=True).unsqueeze(1) # (B*N, 1)
        patch_cluster_assignment_logits = patch_cluster_intersections_volume
        patch_cluster_assignment_probs = F.softmax(patch_cluster_assignment_logits, dim = -1)
        patch_cluster_volume_diff = F.relu(encoded_patch_volume - patch_cluster_intersections_volume)
        patch_cluster_pull_loss = torch.sum(patch_cluster_assignment_probs.detach() * patch_cluster_volume_diff, dim=1).mean()

        full_image_cluster_intersections = self.compute_cluster_intersections(full_image_box_dists, full_image_prior_box_dists) # Internal parameter shape: (B, N_FI, D)
        full_image_cluster_intersections_volume = bessel_volume(full_image_cluster_intersections, volume_temp=0.1, log_scale=True) # (B, N_FI)
        encoded_full_image_volume = bessel_volume(full_image_box_dists, volume_temp=0.1, log_scale=True).unsqueeze(1) # (B, 1)
        full_image_cluster_assignment_logits = full_image_cluster_intersections_volume
        full_image_cluster_assignment_probs = F.softmax(full_image_cluster_assignment_logits, dim = -1)
        full_image_volume_diff = F.relu(encoded_full_image_volume - full_image_cluster_intersections_volume)
        full_image_cluster_pull_loss = torch.sum(full_image_cluster_assignment_probs.detach() * full_image_volume_diff, dim=1).mean()


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
        
        # We use ReLU because numerical noise might make intersection slightly larger, 
        # but theoretically Vol(Subset) <= Vol(Set).
        inclusion_loss = F.relu(log_vol_img - log_vol_intersection).mean()


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


        # ----- Regularization Losses -----
        
        # --- Uniform Prior Loss ---
        loss_uniform_patch = self.compute_marginal_entropy_loss(patch_cluster_assignment_probs)
        loss_uniform_full = self.compute_marginal_entropy_loss(full_image_cluster_assignment_probs)

        # --- Disjoint Prior Loss ---
        loss_disjoint_full = self.compute_pairwise_disjoint_loss(prior_box_dists[1])



        total_loss = (
            self.loss_weights["patch_pull_loss"] * patch_cluster_pull_loss +
            self.loss_weights["full_image_pull_loss"] * full_image_cluster_pull_loss +
            self.loss_weights["inclusion_loss"] * inclusion_loss +
            self.loss_weights["patch_mse_loss"] * recon_loss_patch_mse +
            self.loss_weights["patch_ssim_loss"] * recon_loss_patch_ssim +
            self.loss_weights["full_image_mse_loss"] * recon_loss_full_mse +
            self.loss_weights["full_image_ssim_loss"] * recon_loss_full_ssim +
            self.loss_weights["loss_uniform_patch"] * loss_uniform_patch + 
            self.loss_weights["loss_uniform_full"] * loss_uniform_full + 
            self.loss_weights["loss_disjoint_full"] * loss_disjoint_full
        )

        loss_dict = {
            "total_loss": total_loss,
            "patch_pull_loss": patch_cluster_pull_loss,
            "full_image_pull_loss": full_image_cluster_pull_loss,
            "inclusion_loss": inclusion_loss,
            "patch_mse_loss": recon_loss_patch_mse,
            "patch_ssim_loss": recon_loss_patch_ssim,
            "full_image_mse_loss": recon_loss_full_mse,
            "full_image_ssim_loss": recon_loss_full_ssim,
            "loss_uniform_patch": loss_uniform_patch,
            "loss_uniform_full": loss_uniform_full,
            "loss_disjoint_full": loss_disjoint_full
        }
        self.log_dict(loss_dict, prog_bar=True, on_epoch=True)

        # # ----- PULL LOSS COMPUTATION ----- 
        # patch_log_vol = soft_volume(patch_box_dists, log_scale=True).unsqueeze(1) # (B*N, 1)
        # patch_prior_log_vol = soft_volume(patch_prior_box_dists, log_scale = True).unsqueeze(0) # (1, M)
        # patch_intersection_box_dists = pairwise_gumbel_intersection(patch_box_dists, patch_prior_box_dists)
        # patch_intersection_log_vol = soft_volume(patch_intersection_box_dists, log_scale = True) # (B*N, M)

        # patch_log_likelihoods = patch_intersection_log_vol - patch_log_vol# - patch_prior_log_vol
        # patch_assignment_weights = F.softmax(patch_log_likelihoods, dim = 1) # (B*N, M)
        # patch_assignment_weights = patch_assignment_weights.detach()

        # patch_containment_log_ratio = patch_intersection_log_vol - patch_log_vol # (B*N, M)
        # patch_pull_loss = torch.mean(-torch.sum(patch_assignment_weights * patch_containment_log_ratio, dim=1))


        # full_image_log_vol = soft_volume(full_image_box_dists, log_scale=True).unsqueeze(1)
        # full_image_prior_log_vol = soft_volume(full_image_prior_box_dists, log_scale=True).unsqueeze(0)
        # full_intersection_box_dists = pairwise_gumbel_intersection(full_image_box_dists, full_image_prior_box_dists)
        # full_intersection_log_vol = soft_volume(full_intersection_box_dists, log_scale=True)

        # full_log_likelihoods = full_intersection_log_vol - full_image_log_vol# - full_image_prior_log_vol
        # full_assignment_weights = F.softmax(full_log_likelihoods, dim=1) # (B, K)
        # full_assignment_weights = full_assignment_weights.detach()

        # full_containment_log_ratio = full_intersection_log_vol - full_image_log_vol
        # full_image_pull_loss = torch.mean(-torch.sum(full_assignment_weights * full_containment_log_ratio, dim=1))

        # total_pull_loss = patch_pull_loss + full_image_pull_loss

        
        # # ----- Assignment Weights Regularization -----
        # patch_avg_usage = torch.mean(patch_assignment_weights, dim=0)
        # patch_entropy_loss = torch.sum(patch_avg_usage * torch.log(patch_avg_usage + 1e-20))
        
        # full_avg_usage = torch.mean(full_assignment_weights, dim=0)
        # full_entropy_loss = torch.sum(full_avg_usage * torch.log(full_avg_usage + 1e-20))
        
        # total_entropy_loss = patch_entropy_loss + full_entropy_loss


        # # ----- Consistency Loss -----
        # B = full_image_box_dists.mu_min.shape[0]
        # N = patch_box_dists.mu_min.shape[0] // B
        # D = patch_box_dists.mu_min.shape[-1]
        
        # # def reshape_dist(dist, b, n, d):
        # #     return BoxEmbeddingDistribution(
        # #         dist.mu_min.view(b, n, d).permute(1, 0, 2),
        # #         dist.mu_max.view(b, n, d).permute(1, 0, 2),
        # #         dist.beta_min.view(b, n, d).permute(1, 0, 2),
        # #         dist.beta_max.view(b, n, d).permute(1, 0, 2)
        # #     )

        # # reshaped_patches = reshape_dist(patch_box_dists, B, N, D)
        
        # # patch_weights = torch.ones(N, B, 1, device=self.device)
        # # combined_patch_dist = soft_intersection(reshaped_patches, patch_weights)
        
        # # consistency_int_dist = elementwise_gumbel_intersection(full_image_box_dists, combined_patch_dist)
        
        # # log_vol_full = soft_volume(full_image_box_dists, log_scale=True) # (B,)
        # # log_vol_consistency_int = soft_volume(consistency_int_dist, log_scale=True) # (B,)

        # # consistency_log_ratio = log_vol_consistency_int - log_vol_full
        # # consistency_loss = torch.mean(-consistency_log_ratio)

        # full_mu_min_exp = full_image_box_dists.mu_min.repeat_interleave(N, dim=0)
        # full_mu_max_exp = full_image_box_dists.mu_max.repeat_interleave(N, dim=0)
        # full_beta_min_exp = full_image_box_dists.beta_min.repeat_interleave(N, dim=0)
        # full_beta_max_exp = full_image_box_dists.beta_max.repeat_interleave(N, dim=0)
        
        # expanded_full_dist = BoxEmbeddingDistribution(
        #     full_mu_min_exp, full_mu_max_exp, full_beta_min_exp, full_beta_max_exp
        # )

        # # 2. Intersect "Patch" with "Full Image"
        # # If Patch is inside Full Image, the Intersection should be exactly the Patch.
        # consistency_int_dist = pairwise_gumbel_intersection(patch_box_dists, expanded_full_dist)
        # # Note: pairwise returns (A, B, D), but here we want elementwise since indices are aligned.
        # # But wait, pairwise computes A x B. We just want elementwise (Patch_i with Full_i_expanded).
        # consistency_int_dist = elementwise_gumbel_intersection(patch_box_dists, expanded_full_dist)

        # # 3. Containment Ratio
        # # We want log(Vol(Intersection)) approx equals log(Vol(Patch))
        # log_vol_patch = soft_volume(patch_box_dists, log_scale=True)
        # log_vol_int = soft_volume(consistency_int_dist, log_scale=True)

        # # Minimize the difference (Negative Log Ratio)
        # consistency_log_ratio = log_vol_int - log_vol_patch
        # consistency_loss = torch.mean(-consistency_log_ratio)

        # # ----- Reconstruction Loss -----
        # recon_loss_full_mse = F.mse_loss(full_image_reconstruction, images)
        # recon_loss_full_ssim = 1 - self.ssim_loss_fn(full_image_reconstruction, images)
        # recon_loss_full = recon_loss_full_mse*10 + recon_loss_full_ssim
        # recon_loss_full = recon_loss_full * 10

        # patches = self.divide_image_into_patches(images) # (B, N, C, Hp, Wp)
        # B, N, C, Hp, Wp = patches.shape
        # patches_flat = patches.view(B * N, C, Hp, Wp)
        
        # # Resize targets to VAE resolution (e.g. 64x64)
        # patches_target = F.interpolate(
        #     patches_flat, 
        #     size=self.input_resolution, 
        #     mode='bilinear', 
        #     align_corners=False
        # )

        # recon_loss_patch_mse = F.mse_loss(patch_reconstruction, patches_target)
        # recon_loss_patch_ssim = 1 - self.ssim_loss_fn(patch_reconstruction, patches_target)
        # recon_loss_patch = recon_loss_patch_mse*10 + recon_loss_patch_ssim

        # total_recon_loss = recon_loss_full + recon_loss_patch

        # weights = self.loss_weights
        # total_loss = (weights["reconstruction"] * total_recon_loss) + \
        #              (weights["pull"] * total_pull_loss) + \
        #              (weights["entropy"] * total_entropy_loss) + \
        #              (weights["consistency"] * consistency_loss)

        # # self.log("train/loss", total_loss)
        # # self.log("train/recon_loss", total_recon_loss)
        # # self.log("train/pull_loss", total_pull_loss)
        # # self.log("train/entropy_loss", total_entropy_loss)
        # # self.log("train/consistency_loss", consistency_loss)

        # loss_dict = {
        #     "total_loss": total_loss,
        #     "total_recon_loss": total_recon_loss,
        #     "total_pull_loss": total_pull_loss,
        #     "total_entropy_loss": total_entropy_loss,
        #     "consistency_loss": consistency_loss
        # }

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
        """
        Encourages the batch to utilize all clusters equally (Uniform Prior).
        Args:
            assignment_probs: (Batch, Num_Clusters) - Softmaxed probabilities p(c|x)
        """
        # 1. Compute Marginal Probability (Average usage across batch)
        # Shape: (Num_Clusters,)
        avg_probs = assignment_probs.mean(dim=0)
        
        # 2. Compute KL Divergence to Uniform
        # Target is 1/K for all clusters
        num_clusters = avg_probs.shape[0]
        target_probs = torch.ones_like(avg_probs) / num_clusters
        
        # KL(P || Q) = sum(p * log(p/q))
        # We add epsilon for numerical stability
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
        
        return disjoint_loss

    def divide_image_into_patches(self, images):
        B, C, H, W = images.shape
        grid_h, grid_w = self.grid_size
        patch_h = H // grid_h
        patch_w = W // grid_w

        if H % grid_h != 0 or W % grid_w != 0:
            raise ValueError("Image dimensions must be divisible by grid size.")

        # 1. Split H and W dimensions
        patches = images.view(
            B, C, grid_h, patch_h, grid_w, patch_w
        ) 
        
        # 2. Permute to group grid dimensions and channel/spatial dims
        # (B, grid_h, grid_w, C, patch_h, patch_w)
        patches = patches.permute(0, 2, 4, 1, 3, 5)  
        
        # 3. Collapse grid_h and grid_w into 'Num_Patches' (N)
        # Final Shape: (B, N, C, patch_h, patch_w)
        patches = patches.reshape(B, -1, C, patch_h, patch_w)
        
        return patches

    def configure_optimizers(self):

        if self.config["trainer"]["optimizer"]["type"] == "Adam":
            return torch.optim.Adam(
                self.parameters(),
                **self.config["trainer"]["optimizer"]["config"],
            )
        else:
            raise ValueError(
                f"Optimizer type {self.config['trainer']['optimizer']['type']} not implemented."
            )

    def on_train_epoch_end(self):
        if self.viz_datapoint is not None:
            # Create the figure
            fig = self.create_reconstruction_figure(
                self.viz_datapoint, 
                self.grid_size, 
                self.current_epoch
            )
            
            # Log to WandB (or whatever logger is being used)
            if self.logger is not None:
                # Check if using WandB
                if hasattr(self.logger.experiment, "log"):
                     self.logger.experiment.log({"reconstruction_visualization": fig})
                # Check if using TensorBoard
                elif hasattr(self.logger.experiment, "add_figure"):
                    self.logger.experiment.add_figure(
                        "reconstruction_visualization", fig, self.current_epoch
                    )
            
            # Cleanup
            self.viz_datapoint = None
            plt.close(fig)

    @staticmethod
    def create_reconstruction_figure(data, grid_size, epoch):
        """
        Creates a matplotlib figure comparing original and reconstructed images/patches.
        """
        original_img = data["original_image"].permute(1, 2, 0).numpy()
        recon_img = data["recon_image"].permute(1, 2, 0).numpy()
        
        input_patches = data["input_patches"]
        recon_patches = data["recon_patches"]
        
        # Unnormalize if your images are [-1, 1], otherwise assume [0, 1]
        # Assuming [0, 1] based on Sigmoid in decoder
        
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
        # Create a grid of images
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





# class HierarchicalBoxEmbeddingsVAE(L.LightningModule):
#     def __init__(self, config):
#         super(HierarchicalBoxEmbeddingsVAE, self).__init__()
#         self.config = config

#         self.embed_dim = config["model"]["config"]["embed_dim"]
#         self.hidden_dims = config["model"]["config"]["hidden_dims"]
#         self.boxes_per_level = config["model"]["config"]["boxes_per_level"]
#         self.grid_size = config["model"]["config"]["grid_size"]
#         self.image_size = config["data"]["train"]["config"]["image_size"]
#         self.gumbel_temp = config["model"]["config"]["gumbel_temp"]

#         self.vae = BoxEmbedVAE(self.embed_dim, self.hidden_dims)

#         self.prior = HierarchicalPrior(
#             boxes_per_level=self.boxes_per_level, embed_dim=self.embed_dim, temp=self.gumbel_temp
#         )

#         self.loss_weights = config["model"]["config"]["loss_weights"]
#         self.min_side_length = config["model"]["config"]["min_side_length"]

#         if self.loss_weights.get("ssim", 0) > 0:
#             self.ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3)

#     def divide_into_patches(self, images):
#         B, C, H, W = images.shape
#         gh, gw = self.grid_size
#         ph, pw = H // gh, W // gw
        
#         patches = images.view(B, C, gh, ph, gw, pw) # (B, C, grid_h, patch_h, grid_w, patch_w)
#         patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous() # (B, grid_h, grid_w, C, patch_h, patch_w)
        
#         return patches.view(-1, C, ph, pw) # (B * Num_Patches, C, patch_h, patch_w)

#     def forward(self, x):
#         images = x["images"]
#         B = images.shape[0]
#         num_patches = self.grid_size[0] * self.grid_size[1]

#         patches_flat = self.divide_into_patches(images)
#         patches_resized = F.interpolate(patches_flat, size=self.image_size, mode='bilinear')

#         all_inputs = torch.cat([images, patches_resized], dim=0)
#         vae_out = self.vae(all_inputs)
#         all_boxes = vae_out["box_embeddings"]

#         priors, adj_matrices = self.prior()

#         full_img_boxes = BoxTensor(all_boxes.min_embed[:B], all_boxes.max_embed[:B])
#         patch_boxes = BoxTensor(all_boxes.min_embed[B:], all_boxes.max_embed[B:])

#         return {
#             "vae_out": vae_out,
#             "full_img_boxes": full_img_boxes,
#             "patch_boxes": patch_boxes,
#             "prior_boxes": priors,
#             "adj_matrices": adj_matrices,
#             "inputs": all_inputs
#         }

#     def compute_inclusion_loss(self, encoded_boxes: BoxTensor, prior_boxes: BoxTensor):
#         enc_min = encoded_boxes.min_embed.unsqueeze(1)
#         enc_max = encoded_boxes.max_embed.unsqueeze(1)
#         b_enc = BoxTensor(enc_min, enc_max)
        
#         # prior: (Num_Priors, D) -> (1, Num_Priors, D)
#         pri_min = prior_boxes.min_embed.unsqueeze(0)
#         pri_max = prior_boxes.max_embed.unsqueeze(0)
#         b_pri = BoxTensor(pri_min, pri_max)
        
#         # Intersection: (Batch, Num_Priors, D)
#         # Use prior temp
#         temp = self.prior.intersection_temp
#         inter = gumbel_intersection(b_enc, b_pri, temp)
        
#         # Volume: (Batch, Num_Priors)
#         # Using soft volume
#         vol = soft_volume(inter, temp) # Log volume
        
#         # Softmax over priors to get P(z | prior_k)
#         # We use the volume as the logit. Larger intersection = higher prob.
#         probs = F.softmax(vol, dim=1) # (Batch, Num_Priors)
        
#         # Uniformity Regularization (KL Divergence with Uniform)
#         # We want the batch to use all priors roughly equally
#         avg_probs = probs.mean(dim=0) # (Num_Priors)
#         target_uniform = torch.ones_like(avg_probs) / avg_probs.shape[0]
#         kl_uniform = F.kl_div(avg_probs.log(), target_uniform, reduction='sum')
        
#         # Inclusion Loss: Maximize expected intersection volume
#         # We want to maximize sum(prob * volume). Since vol is log_vol, this is reasonable.
#         # Minimizing negative weighted sum.
#         inclusion = -(probs * vol).sum(dim=1).mean()
        
#         return inclusion, kl_uniform

#     def training_step(self, batch, batch_idx):
#         outputs = self(batch)
        
#         recons = outputs["vae_out"]["reconstruction"]
#         inputs = outputs["inputs"]
#         full_img_boxes = outputs["full_img_boxes"]
#         patch_boxes = outputs["patch_boxes"]
        
#         prior_img = outputs["prior_boxes"][0]
#         prior_patch = outputs["prior_boxes"][1]
        
#         recon_loss = F.mse_loss(recons, inputs)
        
#         if self.loss_weights.get("lpips", 0) > 0:
#             recon_loss += self.lpips_fn(2*recons-1, 2*inputs-1).mean() * self.loss_weights["lpips"]
            
#         if self.loss_weights.get("ssim", 0) > 0:
#             recon_loss += (1 - self.ssim_fn(recons, inputs)) * self.loss_weights["ssim"]

#         incl_img, kl_img = self.compute_inclusion_loss(full_img_boxes, prior_img)
#         incl_patch, kl_patch = self.compute_inclusion_loss(patch_boxes, prior_patch)
        
#         inclusion_total = (incl_img + incl_patch)
#         kl_uniform_total = (kl_img + kl_patch)

#         # --- 3. Regularization ---
#         # Penalize huge boxes (standard box embedding reg)
#         # We penalize both the encoded boxes and the learnable priors
#         # all_boxes_list = [full_img_boxes, patch_boxes, prior_img, prior_patch]
#         # vol_reg = 0
#         # for b in all_boxes_list:
#         #     vol_reg += soft_volume(b, 1.0).mean() # Simple log volume mean

#         # Min side length (Prevent collapse)
#         # side_reg = 0
#         # for b in all_boxes_list:
#         #      sides = b.side_lengths()
#         #      # Penalize if side < min_len
#         #      side_reg += F.relu(self.min_side_length - sides).sum(dim=-1).mean()

#         # --- Total Loss ---
#         total_loss = (
#             recon_loss * self.loss_weights["reconstruction"] +
#             inclusion_total * self.loss_weights["inclusion"] +
#             kl_uniform_total * self.loss_weights["kl_uniform"]
#             # vol_reg * self.loss_weights["volume_reg"] + 
#             # side_reg # Usually weighted heavily or hard constraint
#         )

#         # --- Logging ---
#         self.log_dict({
#             "train/total_loss": total_loss,
#             "train/recon_loss": recon_loss,
#             "train/inclusion_loss": inclusion_total,
#             "train/kl_uniform": kl_uniform_total,
#             # "train/vol_reg": vol_reg,
#         }, prog_bar=True)
        
#         # Visualization Hook
#         if batch_idx == 0 and self.current_epoch % 5 == 0:
#              self.viz_datapoint = {
#                  "inputs": inputs[:4].detach().cpu(),
#                  "recons": recons[:4].detach().cpu()
#              }

#         return total_loss

#     def configure_optimizers(self):

#         if self.config["trainer"]["optimizer"]["type"] == "Adam":
#             return torch.optim.Adam(
#                 self.parameters(),
#                 **self.config["trainer"]["optimizer"]["config"],
#             )
#         else:
#             raise ValueError(
#                 f"Optimizer type {self.config['trainer']['optimizer']['type']} not implemented."
#             )

    
