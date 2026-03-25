import itertools
import random
import io

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import wandb
import lightning as L
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns

from .PerceptualEncoder import ViTEncoder
from .ConceptEncoder import ConceptEncoder, RepresentationCombiner
from .Decoder import ViTDecoder
from .concept_utils import get_inclusion, ridge_projector, get_binary_inclusion

class SubspaceConceptLattice(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config["model"]["config"]

        self.embed_dim = self.model_config["embed_dim"]
        self.ambient_dim = self.model_config["ambient_dim"]
        self.n_attr = self.model_config["n_attr"]
        self.n_obj = self.model_config["n_obj"]
        self.lbd = self.model_config["lbd"]
        self.image_size = self.model_config["image_size"]
        self.image_channels = self.model_config["image_channels"]

        self.max_combinations_per_cardinality = self.model_config["max_combinations_per_cardinality"]
        self.intersection_power_steps = self.model_config["intersection_power_steps"]
        self.intersection_consistency_only_pairs = self.model_config["intersection_consistency_only_pairs"]
        self.binary_intersection = self.model_config.get("binary_intersection", False)
        self.binary_inclusion = self.model_config.get("binary_inclusion", False)
        self.attr_force_orthogonal_basis = self.model_config["attr_force_orthogonal_basis"]
        self.global_galois_loss_start_epoch = self.model_config["global_galois_loss_start_epoch"]
        self.subspace_gumbel_sigmoid = self.model_config["subspace_gumbel_sigmoid"]
        
        self.enable_intermediate_representations = self.model_config.get("enable_intermediate_representations", False)
        self.fillers_per_slot = self.model_config.get("fillers_per_slot", None)
        self.dims_per_slot = self.model_config.get("dims_per_slot", None)

        self.loss_weights = self.model_config["loss_weights"]

        if self.model_config["perceptual_encoder"]["type"] == "ViTEncoder":
            perceptual_encoder_config = self.model_config["perceptual_encoder"]["config"]
            perceptual_encoder_config.update({
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
                "image_channels": self.image_channels,
            })
            self.perceptual_encoder = ViTEncoder(perceptual_encoder_config)

        if self.enable_intermediate_representations:
            num_patches = (self.image_size // self.model_config["perceptual_encoder"]["config"]["patch_size"]) ** 2
            combiner_config = self.model_config.get("representation_combiner", {})
            combiner_config.update({
                "embed_dim": self.embed_dim,
                "num_patches": num_patches,
                "heads": combiner_config.get("heads", self.model_config["concept_encoder"]["config"].get("heads", 8)),
            })
            self.representation_combiner = RepresentationCombiner(combiner_config)

        if self.model_config["concept_encoder"]["type"] == "ConceptEncoder":
            concept_encoder_config = self.model_config["concept_encoder"]["config"]
            concept_encoder_config.update({
                "embed_dim": self.embed_dim,
                "ambient_dim": self.ambient_dim,
                "n_attr": self.n_attr,
                "n_obj": self.n_obj,
                "lbd": self.lbd,
                "attr_force_orthogonal_basis": self.attr_force_orthogonal_basis,
                "subspace_gumbel_sigmoid": self.subspace_gumbel_sigmoid,
                "gumbel_sigmoid_annealing": self.model_config.get("gumbel_sigmoid_annealing", False),
                "gumbel_sigmoid_annealing_start_epoch": self.model_config.get("gumbel_sigmoid_annealing_start_epoch", 0),
                "gumbel_sigmoid_annealing_end_epoch": self.model_config.get("gumbel_sigmoid_annealing_end_epoch", 0),
                "gumbel_sigmoid_annealing_start_temp": self.model_config.get("gumbel_sigmoid_annealing_start_temp", 1.0),
                "gumbel_sigmoid_annealing_end_temp": self.model_config.get("gumbel_sigmoid_annealing_end_temp", 0.1),
                "gumbel_sigmoid_hard": self.model_config.get("gumbel_sigmoid_hard", False),
                "fillers_per_slot": self.fillers_per_slot,
                "dims_per_slot": self.dims_per_slot,
            })
            self.concept_encoder = ConceptEncoder(concept_encoder_config)
        
        if self.model_config["decoder"]["type"] == "ViTDecoder":
            decoder_config = self.model_config["decoder"]["config"]
            decoder_config.update({
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
                "image_channels": self.image_channels,
            })
            self.decoder = ViTDecoder(decoder_config)

        self.viz_datapoint = None

    def forward(self, x):
        images = x["images"] # (B, 3, H, W)
        B = images.shape[0]
        representations = self.perceptual_encoder(images) # (B, num_patches, embed_dim)

        all_X_attr = []
        all_X_obj = []
        all_P_attr = []
        all_P_obj = []
        all_cardinalities = []
        all_combination_indices = []

        for k in range(1, B + 1):
            combination_indices = list(itertools.combinations(range(B), k))
            if len(combination_indices) > self.max_combinations_per_cardinality and k > 1:
                combination_indices = random.sample(combination_indices, self.max_combinations_per_cardinality)
            
            combination_indices_tensor = torch.tensor(combination_indices, device=self.device) # (num_combinations, k)
            R_combination_subset = representations[combination_indices_tensor] # (num_combinations, k, num_patches, embed_dim)

            if self.enable_intermediate_representations:
                # Combine multiple representations into a single intermediate representation
                R_combination_subset = self.representation_combiner(R_combination_subset) # (num_combinations, 1, num_patches, embed_dim)

            X_attr, X_obj, P_attr, P_obj = self.concept_encoder(R_combination_subset, current_epoch=getattr(self, 'current_epoch', None))

            all_X_attr.append(X_attr) # (num_combinations, n_attr, d_ambient)
            all_X_obj.append(X_obj) # (num_combinations, n_obj, d_ambient)
            all_P_attr.append(P_attr) # (num_combinations, d_ambient, d_ambient)
            all_P_obj.append(P_obj) # (num_combinations, d_ambient, d_ambient)
            all_cardinalities.append(torch.tensor([k] * len(combination_indices), device=self.device)) # (num_combinations,)
            all_combination_indices.extend(combination_indices_tensor) # (num_combinations, k)

        X_attr_tensor = torch.cat(all_X_attr, dim=0) # (B_total, n_attr, d_ambient)
        X_obj_tensor = torch.cat(all_X_obj, dim=0) # (B_total, n_obj, d_ambient)
        P_attr_tensor = torch.cat(all_P_attr, dim=0) # (B_total, d_ambient, d_ambient)
        P_obj_tensor = torch.cat(all_P_obj, dim=0) # (B_total, d_ambient, d_ambient)
        cardinalities_tensor = torch.cat(all_cardinalities, dim=0) # (B_total,)

        singletons_mask = (cardinalities_tensor == 1)
        X_attr_singletons = X_attr_tensor[singletons_mask] # (B, n_attr, d_ambient)
        X_attr_singletons_dec = self.concept_encoder.attr_dec(X_attr_singletons) # (B, n_attr, embed_dim)

        reconstructed_images = self.decoder(X_attr_singletons_dec) # (B, 3, H, W)

        return {
            "images": images,
            "reconstructed_images": reconstructed_images,
            "X_attr_tensor": X_attr_tensor,
            "X_obj_tensor": X_obj_tensor,
            "P_attr_tensor": P_attr_tensor,
            "P_obj_tensor": P_obj_tensor,
            "cardinalities_tensor": cardinalities_tensor,
            "combination_indices": all_combination_indices,
            "singletons_mask": singletons_mask
        }

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        images = outputs["images"] # (B, 3, H, W)
        reconstructed_images = outputs["reconstructed_images"] # (B, 3, H, W)
        B = images.shape[0]

        X_attr_tensor = outputs["X_attr_tensor"] # (B_total, n_attr, ambient_dim)
        X_obj_tensor = outputs["X_obj_tensor"] # (B_total, n_obj, ambient_dim)
        P_attr_tensor = outputs["P_attr_tensor"] # (B_total, ambient_dim, ambient_dim)
        P_obj_tensor = outputs["P_obj_tensor"] # (B_total, ambient_dim, ambient_dim)
        cardinalities_tensor = outputs["cardinalities_tensor"] # (B_total,)
        combination_indices = outputs["combination_indices"]
        singletons_mask = outputs["singletons_mask"]
        B_total = X_attr_tensor.shape[0]

        P_attr_singletons = P_attr_tensor[singletons_mask] # (B, ambient_dim, ambient_dim)
        P_obj_singletons = P_obj_tensor[singletons_mask] # (B, ambient_dim, ambient_dim)
        X_obj_singletons = X_obj_tensor[singletons_mask] # (B, n_obj, ambient_dim)
        X_attr_singletons = X_attr_tensor[singletons_mask] # (B, n_attr, ambient_dim)
        
        if self.viz_datapoint is None:
            num_samples = min(4, B)
            self.viz_datapoint = {
                "original_images": images[:num_samples].detach().cpu(),
                "reconstructed_images": reconstructed_images[:num_samples].detach().cpu(),
            }

        # LOSS COMPUTATIONS

        ## Reconstruction Loss
        reconstruction_loss = F.mse_loss(reconstructed_images, images, reduction="sum") / B

        ## Singleton Object Rank Loss
        singleton_obj_ranks = torch.einsum("bii->b", P_obj_singletons) 
        singleton_obj_rank_loss = F.mse_loss(singleton_obj_ranks, torch.ones_like(singleton_obj_ranks))

        ## Maximize Singleton Attribute Rank Loss
        singleton_attr_ranks = torch.einsum("bii->b", P_attr_singletons) 
        max_singleton_attr_rank_loss = -torch.mean(singleton_attr_ranks)

        ## Attribute Orthogonality Loss
        attr_orthogonality_loss = self.compute_orthogonality_loss(X_attr_singletons)

        ## Attribute Polarization Loss
        attr_polarization_loss = self.compute_polarization_loss(X_attr_singletons)

        ## Concept Similarity Loss
        concept_similarity_loss = self.compute_proportional_similarity_loss(
            P_attr_singletons,
            P_obj_singletons
        )

        ## Union Consistency Loss
        total_union_consistency_loss = 0
        ## Intersection Consistency Loss
        total_intersection_consistency_loss = 0
        ## Modular Subspace Loss
        total_modular_subspace_loss = 0
        ## Galois Loss
        total_galois_attr_loss = 0
        total_galois_obj_loss = 0
        total_comparisons = 0
        ## Atttibute Sink Loss
        total_attr_sink_loss = 0

        for cardinality in range(2, B + 1):
            cardinality_mask = (cardinalities_tensor == cardinality)
            if not cardinality_mask.any():
                continue
            cardinality_combination_indices = torch.stack([combination_indices[i] for i in torch.where(cardinality_mask)[0]], dim=0).to(self.device) # (num_combinations, k)
            num_combinations = cardinality_combination_indices.shape[0]

            cardinality_P_attr = P_attr_tensor[cardinality_mask] # (num_combinations, ambient_dim, ambient_dim)
            cardinality_P_obj = P_obj_tensor[cardinality_mask] # (num_combinations, ambient_dim, ambient_dim)

            multiplier = (cardinality - 1) / max(1, B - 1)
            actual_attr_rank = torch.einsum("bii->b", cardinality_P_attr) # (num_combinations,)
            total_attr_sink_loss += torch.sum((actual_attr_rank * multiplier) ** 2)

            combination_singletons_P_attr = P_attr_singletons[cardinality_combination_indices] # (num_combinations, k, ambient_dim, ambient_dim)
            combination_singletons_P_obj = P_obj_singletons[cardinality_combination_indices] # (num_combinations, k, ambient_dim, ambient_dim)
            
            combination_singletons_X_obj = X_obj_singletons[cardinality_combination_indices] # (num_combinations, k, n_obj, ambient_dim)
            n_obj = combination_singletons_X_obj.shape[2]
            X_obj_union_basis = combination_singletons_X_obj.view(num_combinations, cardinality * n_obj, self.ambient_dim) # (num_combinations, k * n_obj, ambient_dim)
            P_obj_union = ridge_projector(X_obj_union_basis, lbd=self.lbd) # (num_combinations, ambient_dim, ambient_dim)
            total_union_consistency_loss += F.mse_loss(cardinality_P_obj, P_obj_union, reduction="sum")

            if self.binary_intersection:
                combination_singletons_X_attr = X_attr_singletons[cardinality_combination_indices] # (num_combinations, k, n_attr, ambient_dim)
                X_attr_inter = torch.min(combination_singletons_X_attr, dim=1)[0] # (num_combinations, n_attr, ambient_dim)
                cardinality_X_attr = X_attr_tensor[cardinality_mask] # (num_combinations, n_attr, ambient_dim)
                total_intersection_consistency_loss += F.mse_loss(cardinality_X_attr, X_attr_inter, reduction="sum")

                total_modular_subspace_loss += 0
            elif self.intersection_consistency_only_pairs:
                if cardinality == 2:
                    Pi = combination_singletons_P_attr[:, 0] # (num_combinations, ambient_dim, ambient_dim)
                    Pj = combination_singletons_P_attr[:, 1] # (num_combinations, ambient_dim, ambient_dim)
                    combination_singletons_P_attr_inter = 0.5 * (Pi @ Pj + Pj @ Pi) # (num_combinations, ambient_dim, ambient_dim)
                    total_intersection_consistency_loss += F.mse_loss(cardinality_P_attr, combination_singletons_P_attr_inter, reduction="sum")

                    Xi = combination_singletons_X_obj[:, 0] # (num_combinations, n_obj, ambient_dim)
                    Xj = combination_singletons_X_obj[:, 1] # (num_combinations, n_obj, ambient_dim)
                    Xi_union_Xj = torch.cat([Xi, Xj], dim=1) # (num_combinations, 2 * n_obj, ambient_dim)
                    Xi_union_Xj_P_attr = ridge_projector(Xi_union_Xj, lbd=self.lbd) # (num_combinations, ambient_dim, ambient_dim)
                    target_inter_rank = torch.einsum("bii->b", Pi) + torch.einsum("bii->b", Pj) - torch.einsum("bii->b", Xi_union_Xj_P_attr) # (num_combinations,)
                    actual_inter_rank = torch.einsum("bii->b", cardinality_P_attr) # (num_combinations,)
                    total_modular_subspace_loss += F.mse_loss(actual_inter_rank, target_inter_rank, reduction="sum")
            else:
                combination_singletons_P_attr_avg = torch.mean(combination_singletons_P_attr, dim=1) # (num_combinations, ambient_dim, ambient_dim)
                combination_singletons_P_attr_inter = torch.linalg.matrix_power(combination_singletons_P_attr_avg, self.intersection_power_steps) # (num_combinations, ambient_dim, ambient_dim)
                total_intersection_consistency_loss += F.mse_loss(cardinality_P_attr, combination_singletons_P_attr_inter, reduction="sum")

                total_modular_subspace_loss += 0

            P_attr_comb_exp = cardinality_P_attr.unsqueeze(1).expand(-1, cardinality, -1, -1) # (num_combinations, k, ambient_dim, ambient_dim)
            P_obj_comb_exp = cardinality_P_obj.unsqueeze(1).expand(-1, cardinality, -1, -1) # (num_combinations, k, ambient_dim, ambient_dim)

            P_attr_c_flat = P_attr_comb_exp.reshape(-1, self.ambient_dim, self.ambient_dim) # (num_combinations * k, ambient_dim, ambient_dim)
            P_attr_s_flat = combination_singletons_P_attr.reshape(-1, self.ambient_dim, self.ambient_dim) # (num_combinations * k, ambient_dim, ambient_dim)
            P_obj_c_flat = P_obj_comb_exp.reshape(-1, self.ambient_dim, self.ambient_dim) # (num_combinations * k, ambient_dim, ambient_dim)
            P_obj_s_flat = combination_singletons_P_obj.reshape(-1, self.ambient_dim, self.ambient_dim) # (num_combinations * k, ambient_dim, ambient_dim)

            if getattr(self, "binary_inclusion", False):
                cardinality_X_attr_global = X_attr_tensor[cardinality_mask]
                X_attr_comb_exp = cardinality_X_attr_global.unsqueeze(1).expand(-1, cardinality, -1, -1)
                combination_singletons_X_attr_global = X_attr_singletons[cardinality_combination_indices]
                X_attr_c_flat = X_attr_comb_exp.reshape(-1, self.n_attr, self.ambient_dim)
                X_attr_s_flat = combination_singletons_X_attr_global.reshape(-1, self.n_attr, self.ambient_dim)
                galois_attr_inclusion = get_binary_inclusion(X_attr_c_flat, X_attr_s_flat)
            else:
                galois_attr_inclusion = get_inclusion(P_sub=P_attr_c_flat, P_super=P_attr_s_flat) # (num_combinations * k,)
            
            galois_obj_inclusion = get_inclusion(P_sub=P_obj_s_flat, P_super=P_obj_c_flat) # (num_combinations * k,)

            attr_target = torch.ones_like(galois_attr_inclusion) # (num_combinations * k,)
            obj_target = torch.ones_like(galois_obj_inclusion) # (num_combinations * k,)

            total_galois_attr_loss += F.binary_cross_entropy(galois_attr_inclusion, attr_target, reduction="sum")
            total_galois_obj_loss += F.binary_cross_entropy(galois_obj_inclusion, obj_target, reduction="sum")
            total_comparisons += num_combinations * cardinality

        union_consistency_loss = total_union_consistency_loss / max(1, (B_total - B))
        attr_sink_loss = total_attr_sink_loss / max(1, (B_total - B))
        if self.intersection_consistency_only_pairs:
            intersection_consistency_loss = total_intersection_consistency_loss / 10
            modular_subspace_loss = total_modular_subspace_loss / 10
        else:
            intersection_consistency_loss = total_intersection_consistency_loss / max(1, (B_total - B))
            modular_subspace_loss = total_modular_subspace_loss / max(1, (B_total - B))

        galois_attr_loss = total_galois_attr_loss / max(1, total_comparisons)
        galois_obj_loss = total_galois_obj_loss / max(1, total_comparisons)

        if self.global_galois_loss_start_epoch is not None and self.current_epoch >= self.global_galois_loss_start_epoch:
            global_galois_loss = self.compute_global_galois_loss(P_attr_tensor, P_obj_tensor, X_attr_tensor)
        else:
            global_galois_loss = 0

        basis_orthogonality_loss = self.compute_basis_orthogonality_loss(X_attr_singletons)
        basis_sparsity_loss = self.compute_basis_sparsity_loss(X_attr_singletons)

        cardinalities = outputs["cardinalities_tensor"].float()
        obj_ranks = torch.einsum("bii->b", P_obj_tensor)
        attr_ranks = torch.einsum("bii->b", P_attr_tensor)

        ## Proportionality Losses
        ### Object rank is directly proportional to cardinality
        loss_obj_card_prop = self.proportionality_loss(cardinalities, obj_ranks, inverse=False)
        ### Attribute rank is inversely proportional to cardinality
        loss_attr_card_inv_prop = self.proportionality_loss(cardinalities, attr_ranks, inverse=True)
        ### Inverse proportionality across the concept space (attributes vs objects)
        loss_attr_obj_inv_prop = self.proportionality_loss(attr_ranks, obj_ranks, inverse=True)

        ## Repulsion Losses
        repulsion_loss_obj = self.compute_soft_repulsion(P_obj_singletons)
        repulsion_loss_attr = self.compute_soft_repulsion(P_attr_singletons)

        ## Residual Orthogonality and Rank Conservation Losses
        residual_orthogonality_loss, rank_conservation_loss = self.compute_residual_orthogonality_loss(
            P_attr_singletons, X_attr_singletons
        )

        total_loss = (
            self.loss_weights["reconstruction_loss"] * reconstruction_loss +
            self.loss_weights["singleton_obj_rank_loss"] * singleton_obj_rank_loss +
            self.loss_weights["max_singleton_attr_rank_loss"] * max_singleton_attr_rank_loss +
            self.loss_weights["modular_subspace_loss"] * modular_subspace_loss +
            self.loss_weights["attr_orthogonality_loss"] * attr_orthogonality_loss +
            self.loss_weights["attr_polarization_loss"] * attr_polarization_loss +
            self.loss_weights["concept_similarity_loss"] * concept_similarity_loss +
            self.loss_weights["union_consistency_loss"] * union_consistency_loss +
            self.loss_weights["intersection_consistency_loss"] * intersection_consistency_loss +
            self.loss_weights["attr_sink_loss"] * attr_sink_loss +
            self.loss_weights["galois_attr_loss"] * galois_attr_loss +
            self.loss_weights["galois_obj_loss"] * galois_obj_loss +
            self.loss_weights["global_galois_loss"] * global_galois_loss +
            self.loss_weights["basis_orthogonality_loss"] * basis_orthogonality_loss +
            self.loss_weights["loss_obj_card_prop"] * loss_obj_card_prop +
            self.loss_weights["loss_attr_card_inv_prop"] * loss_attr_card_inv_prop +
            self.loss_weights["loss_attr_obj_inv_prop"] * loss_attr_obj_inv_prop +
            self.loss_weights["repulsion_loss_obj"] * repulsion_loss_obj +
            self.loss_weights["repulsion_loss_attr"] * repulsion_loss_attr +
            self.loss_weights.get("residual_orthogonality_loss", 0) * residual_orthogonality_loss +
            self.loss_weights.get("rank_conservation_loss", 0) * rank_conservation_loss +
            self.loss_weights.get("basis_sparsity_loss", 0) * basis_sparsity_loss
        )

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "singleton_obj_rank_loss": singleton_obj_rank_loss,
            "max_singleton_attr_rank_loss": max_singleton_attr_rank_loss,
            "modular_subspace_loss": modular_subspace_loss,
            "attr_orthogonality_loss": attr_orthogonality_loss,
            "attr_polarization_loss": attr_polarization_loss,
            "concept_similarity_loss": concept_similarity_loss,
            "union_consistency_loss": union_consistency_loss,
            "intersection_consistency_loss": intersection_consistency_loss,
            "attr_sink_loss": attr_sink_loss,
            "galois_attr_loss": galois_attr_loss,
            "galois_obj_loss": galois_obj_loss,
            "global_galois_loss": global_galois_loss,
            "basis_orthogonality_loss": basis_orthogonality_loss,
            "loss_obj_card_prop": loss_obj_card_prop,
            "loss_attr_card_inv_prop": loss_attr_card_inv_prop,
            "loss_attr_obj_inv_prop": loss_attr_obj_inv_prop,
            "repulsion_loss_obj": repulsion_loss_obj,
            "repulsion_loss_attr": repulsion_loss_attr,
            "residual_orthogonality_loss": residual_orthogonality_loss,
            "rank_conservation_loss": rank_conservation_loss,
            "basis_sparsity_loss": basis_sparsity_loss,
            "total_loss": total_loss
        }

        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)

        for cardinality in range(1, B + 1):
            mask = (cardinalities_tensor == cardinality)
            if mask.any():
                avg_attr_rank = attr_ranks[mask].mean()
                avg_obj_rank = obj_ranks[mask].mean()
                self.log(f"rank/attr_cardinality_{cardinality}", avg_attr_rank, on_epoch=True, prog_bar=False)
                self.log(f"rank/obj_cardinality_{cardinality}", avg_obj_rank, on_epoch=True, prog_bar=False)

        return total_loss

    def compute_proportional_similarity_loss(self, P_attr, P_obj):
        """
        Enforces that the similarity between attribute subspaces is proportional 
        to the similarity between object subspaces.
        """
        B, D, _ = P_attr.shape
        
        # Vectorize projectors: (B, D*D)           
        P_attr_vec = P_attr.reshape(B, -1)
        P_obj_vec = P_obj.reshape(B, -1)
        
        # Compute pairwise overlaps for the batch: Tr(P_i P_j) = vec(P_i)^T vec(P_j)
        overlap_attr = torch.matmul(P_attr_vec, P_attr_vec.transpose(0, 1)) # (B, B)
        overlap_obj = torch.matmul(P_obj_vec, P_obj_vec.transpose(0, 1)) # (B, B)
        
        # Extract the diagonal (which represents the trace/rank of each individual projector)
        norm_attr = torch.diag(overlap_attr).clamp(min=1e-6) # (B,)
        norm_obj = torch.diag(overlap_obj).clamp(min=1e-6) # (B,)
        
        # Compute the denominator for cosine similarity normalization
        denom_attr = torch.sqrt(norm_attr.unsqueeze(1) * norm_attr.unsqueeze(0)) # (B, B)
        denom_obj = torch.sqrt(norm_obj.unsqueeze(1) * norm_obj.unsqueeze(0)) # (B, B)
        
        # Normalize to [0, 1] to enforce strict proportionality
        sim_attr = overlap_attr / denom_attr # (B, B)
        sim_obj = overlap_obj / denom_obj # (B, B)
        
        # Enforce Sim(S_attr) is proportional to Sim(S_obj)
        sim_loss = F.mse_loss(sim_attr, sim_obj) # Scalar
        
        return sim_loss

    def proportionality_loss(self, x, y, inverse=False, eps=1e-8):
        """
        Enforces direct or inverse proportionality between two 1D tensors.
        """
        # Center the variables
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        # Compute covariance and variances
        covar = (x_centered * y_centered).sum()
        var_x = (x_centered ** 2).sum()
        var_y = (y_centered ** 2).sum()
        
        # Compute Pearson correlation coefficient
        corr = covar / (torch.sqrt(var_x * var_y) + eps)
        
        if inverse:
            # Enforce y \propto -x (Loss is 0 when corr is -1)
            return 1.0 + corr
        else:
            # Enforce y \propto x (Loss is 0 when corr is 1)
            return 1.0 - corr

    def compute_soft_repulsion(self, P_singletons):
        """
        Minimizes the average pairwise overlap of subspaces in the batch.
        True duplicates will resist this via reconstruction and proportionality losses,
        causing only distinct concepts to orthogonalize.
        """
        B = P_singletons.shape[0]
        
        # Vectorize projectors: (B, D*D)
        P_vec = P_singletons.reshape(B, -1)
        
        # Compute pairwise overlap (trace of products): (B, B)
        overlap = torch.matmul(P_vec, P_vec.transpose(0, 1))
        
        # Mask out the diagonal (we don't want to penalize a concept overlapping with itself)
        mask = ~torch.eye(B, dtype=torch.bool, device=P_singletons.device)
        
        # Return the mean of the off-diagonal overlaps
        return overlap[mask].mean()

    def compute_orthogonality_loss(self, X):
        """Encourages basis vectors to be orthogonal, maximizing subspace rank naturally."""
        B, N, D = X.shape
        # Compute Gram matrix
        if self.model_config.get("attr_orthogonality_avg_first", False):
            X_mean = torch.mean(torch.abs(X), dim=0) # (N, D)
            norms = torch.norm(X_mean, dim=-1, keepdim=True) + 1e-8
            X_mean = X_mean / norms
            overlap = torch.mm(X_mean, X_mean.t()) # (N, N)
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
            return (overlap[mask] ** 2).mean()

        else:
            gram = torch.bmm(X, X.transpose(1, 2)) # (B, N, N)
            
            # Normalize to get cosine similarities
            norms = torch.norm(X, dim=-1, keepdim=True) + 1e-8
            gram_norm = gram / torch.bmm(norms, norms.transpose(1, 2))
            
            # Mask out the diagonal
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device).expand(B, N, N)
            
            # Penalize off-diagonal correlations
            return (gram_norm[mask] ** 2).mean()

    def compute_polarization_loss(self, X):
        B, N, D = X.shape
        polarization_loss = 0.0
        for basis_idx in range(N):
            basis_vectors = X[:, basis_idx, :] # (B, D)
            norm = F.normalize(basis_vectors, p=2, dim=-1)
            sim_matrix = torch.mm(norm, norm.t()) # (B, B)
            sim_squared = sim_matrix ** 2
            ambiguity_penalty = sim_squared * (1.0 - sim_squared)
            polarization_loss += ambiguity_penalty.mean()
        
        return polarization_loss / N

    def compute_global_galois_loss(self, P_attr_tensor, P_obj_tensor, X_attr_tensor=None, threshold=0.85):
        """
        Discovers empirical inclusions across the entire batch and enforces the Galois connection.
        If S_a^attr is included in S_b^attr (score > threshold), 
        then S_b^obj must be included in S_a^obj (target = 1.0).
        """
        N = P_attr_tensor.shape[0]
        
        if getattr(self, "binary_inclusion", False) and X_attr_tensor is not None:
            X_attr_flat = X_attr_tensor.reshape(N, -1)
            X_diff = X_attr_flat.unsqueeze(1) - X_attr_flat.unsqueeze(0) # (N, N, D)
            inc_attr = 1.0 - torch.mean(F.relu(X_diff), dim=2) # (N, N)
        else:
            # 1. Flatten projectors: (N, D_ambient * D_ambient)
            P_attr_flat = P_attr_tensor.reshape(N, -1)
            # 2. Compute all pairwise overlaps: Tr(P_i P_j) -> shape (N, N)
            overlap_attr = torch.matmul(P_attr_flat, P_attr_flat.t())
            # 3. Compute traces (ranks) for the denominators -> shape (N,)
            trace_attr = torch.einsum("bii->b", P_attr_tensor).clamp(min=1e-6)
            # 4. Compute Inclusion Matrices: I[i, j] = P(i is subset of j)
            inc_attr = overlap_attr / trace_attr.unsqueeze(1) # (N, N)

        P_obj_flat = P_obj_tensor.reshape(N, -1)
        overlap_obj = torch.matmul(P_obj_flat, P_obj_flat.t())
        trace_obj = torch.einsum("bii->b", P_obj_tensor).clamp(min=1e-6)
        inc_obj = overlap_obj / trace_obj.unsqueeze(1)    # (N, N)
        
        # Clamp to avoid BCE boundary instability
        inc_attr = torch.clamp(inc_attr, 1e-6, 1.0 - 1e-6)
        inc_obj = torch.clamp(inc_obj, 1e-6, 1.0 - 1e-6)
        
        # 5. Create discovery masks
        # .detach() is CRITICAL here so the thresholding doesn't create gradient feedback loops
        mask_attr = (inc_attr > threshold).detach()
        mask_obj = (inc_obj > threshold).detach()
        
        # Remove the diagonal (a concept always includes itself, no loss needed)
        mask_attr.fill_diagonal_(False)
        mask_obj.fill_diagonal_(False)
        
        # 6. Apply Galois logic via transposed indexing
        # Forward Galois: If Attr_i in Attr_j (mask_attr[i,j]), we want Obj_j in Obj_i (inc_obj[j,i])
        inc_obj_target_preds = inc_obj.t()[mask_attr] 
        
        # Reverse Galois: If Obj_i in Obj_j (mask_obj[i,j]), we want Attr_j in Attr_i (inc_attr[j,i])
        inc_attr_target_preds = inc_attr.t()[mask_obj]
        
        loss_forward = 0.0
        loss_reverse = 0.0
        
        # 7. Compute BCE losses if any inclusions were discovered
        if inc_obj_target_preds.numel() > 0:
            loss_forward = F.binary_cross_entropy(
                inc_obj_target_preds, 
                torch.ones_like(inc_obj_target_preds)
            )
            
        if inc_attr_target_preds.numel() > 0:
            loss_reverse = F.binary_cross_entropy(
                inc_attr_target_preds, 
                torch.ones_like(inc_attr_target_preds)
            )
            
        return loss_forward + loss_reverse

    def compute_residual_orthogonality_loss(self, P_attr_singletons, X_attr_singletons):
        """
        For each pair of singleton attr subspaces, compute the intersection and the
        residual (complement within each original subspace). Then enforce:
        1. Residual orthogonality: the residuals from the two concepts should be orthogonal
        2. Rank conservation: intersection_rank + residual_rank = original_rank

        Operates on projectors for non-binary mode, or on basis vectors for binary mode.
        Returns (residual_orthogonality_loss, rank_conservation_loss).
        """
        B = P_attr_singletons.shape[0]
        if B < 2:
            zero = torch.tensor(0.0, device=P_attr_singletons.device)
            return zero, zero

        total_ortho_loss = 0.0
        total_conservation_loss = 0.0
        num_pairs = 0

        for i in range(B):
            for j in range(i + 1, B):
                P_i = P_attr_singletons[i]  # (D, D)
                P_j = P_attr_singletons[j]  # (D, D)

                if self.binary_intersection:
                    # Binary mode: operate on basis vectors X_attr
                    X_i = X_attr_singletons[i]  # (n_attr, D)
                    X_j = X_attr_singletons[j]  # (n_attr, D)

                    # Intersection: element-wise min (binary intersection)
                    X_inter = torch.min(X_i, X_j)  # (n_attr, D)

                    # Residual: what's in each but not in the intersection
                    X_res_i = F.relu(X_i - X_inter)  # (n_attr, D)
                    X_res_j = F.relu(X_j - X_inter)  # (n_attr, D)

                    # Compute projectors from residuals for orthogonality check
                    P_res_i = ridge_projector(X_res_i.unsqueeze(0), lbd=self.lbd).squeeze(0)  # (D, D)
                    P_res_j = ridge_projector(X_res_j.unsqueeze(0), lbd=self.lbd).squeeze(0)  # (D, D)
                    P_inter = ridge_projector(X_inter.unsqueeze(0), lbd=self.lbd).squeeze(0)  # (D, D)
                else:
                    # Projector mode: intersection via alternating projection
                    P_inter = 0.5 * (P_i @ P_j + P_j @ P_i)  # (D, D)

                    # Residuals: complement of intersection within each subspace
                    P_res_i = P_i - P_inter  # (D, D)
                    P_res_j = P_j - P_inter  # (D, D)

                # Residual orthogonality: trace(P_res_i @ P_res_j) should be 0
                overlap = torch.trace(P_res_i @ P_res_j)
                total_ortho_loss += overlap ** 2

                # Rank conservation: rank(intersection) + rank(residual) = rank(original)
                rank_i = torch.trace(P_i)
                rank_inter = torch.trace(P_inter)
                rank_res_i = torch.trace(P_res_i)
                total_conservation_loss += (rank_inter + rank_res_i - rank_i) ** 2

                rank_j = torch.trace(P_j)
                rank_res_j = torch.trace(P_res_j)
                total_conservation_loss += (rank_inter + rank_res_j - rank_j) ** 2

                num_pairs += 1

        if num_pairs > 0:
            total_ortho_loss = total_ortho_loss / num_pairs
            total_conservation_loss = total_conservation_loss / num_pairs

        return total_ortho_loss, total_conservation_loss

    def compute_basis_sparsity_loss(self, X_singletons):
        """
        Encourages each singleton's basis vector to have exactly one active dimension
        per basis slot (one-hot). For each basis slot, the L1 norm of the singleton's
        basis vector should be 1.0 (exactly one dim on, rest off).

        Args:
            X_singletons: (B, n_attr, ambient_dim) — singleton attr basis vectors
        Returns:
            Scalar loss
        """
        B, N, D = X_singletons.shape
        loss = 0.0
        for basis_idx in range(N):
            basis_vectors = X_singletons[:, basis_idx, :]  # (B, D)
            l1_norms = basis_vectors.abs().sum(dim=-1)  # (B,)
            loss += F.mse_loss(l1_norms, torch.ones_like(l1_norms))
        return loss / N

    def compute_basis_orthogonality_loss(self, X):
        B, N, D = X.shape

        loss_rank = 0.0
        loss_ortho = 0.0
        total_loss = 0.0
        basis_orthogonality_loss_type = self.config["model"]["config"]["basis_orthogonality_loss_type"]
        default_target = float(self.config["model"]["config"].get("basis_orthogonality_rank_target", 2.0))

        # Determine per-slot rank targets
        if self.fillers_per_slot is not None:
            dims_per_slot = self.concept_encoder.dims_per_slot
            rank_targets = [min(f, d) for f, d in zip(self.fillers_per_slot, dims_per_slot)]
        else:
            rank_targets = [default_target] * N

        for basis_idx in range(N):
            basis_vectors = X[:, basis_idx, :] # (B, D)
            X_norm = F.normalize(basis_vectors, p=2, dim=-1) # (B, D)
            C = torch.mm(X_norm.t(), X_norm) # (D, D)
            C_ridge = C + self.lbd * torch.eye(D, device=self.device)
            P_cov = torch.mm(C, torch.linalg.inv(C_ridge)) # (D, D)
            current_rank = torch.trace(P_cov)
            target_rank = float(rank_targets[basis_idx])
            loss_rank += F.mse_loss(current_rank, torch.tensor(target_rank, device=self.device))

            G = torch.mm(X_norm, X_norm.t()) # (B, B)
            mask = ~torch.eye(B, dtype=torch.bool, device=self.device)
            loss_ortho += torch.mean(G[mask] ** 2)

        loss_rank /= N
        loss_ortho /= N

        if "rank" in basis_orthogonality_loss_type:
            total_loss += loss_rank
        if "ortho" in basis_orthogonality_loss_type:
            total_loss += loss_ortho

        return total_loss

    @torch.no_grad()
    def log_concept_lattice_inclusion(self):
        """
        Generates ideal images using dataset transforms, forms the formal concepts
        dynamically, and plots heatmaps of their subspace inclusions.
        """
        self.eval()
        
        train_config = self.config["data"]["train"]["config"]
        img_size = train_config.get("image_size", 64)
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            img_size_x, img_size_y = img_size
        else:
            img_size_x = img_size_y = img_size

        shapes = train_config.get("shapes", ["circle", "square"])
        colors = train_config.get("colors", ["red", "blue", "green"])
        
        color_map = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
                     "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
                     "white": (255, 255, 255), "black": (0, 0, 0)}
        
        # 1. Define the exact same transform used in v0Dataset
        default_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        center_val = train_config.get("center_range", [32, 33])[0]
        size_val = train_config.get("size_range", [20, 21])[0]

        def make_ideal_img(shape, color_name):
            # Create the base HWC uint8 numpy array just like generate_image()
            img = np.zeros([img_size_x, img_size_y, 3], dtype=np.uint8)
            color = color_map.get(color_name, (255, 255, 255))
            center = (center_val, center_val)
            
            if shape == "circle":
                radius = size_val // 2
                cv2.circle(img, center, radius, color, -1)
            elif shape == "square":
                side = size_val
                cv2.rectangle(img, 
                            (center[0] - side // 2, center[1] - side // 2), 
                            (center[0] + side // 2, center[1] + side // 2), 
                            color, -1)
                            
            # Apply the default torchvision transform
            img_tensor = default_transform(img)
            
            # Move to the correct device
            return img_tensor.to(self.device)

        # Generate ALL combinations (including excluded) for the lattice visualization
        excluded = set(train_config.get("excluded_combinations", []))
        combinations = list(itertools.product(colors, shapes))

        images = []
        singleton_labels = []
        for c, s in combinations:
            images.append(make_ideal_img(s, c))
            label = f"{c.capitalize()} {s.capitalize()}"
            if f"{c} {s}" in excluded:
                label += " *"
            singleton_labels.append(label)
        
        images = torch.stack(images) # (num_singletons, 3, H, W)
        num_singletons = len(combinations)
        
        # 2. Get perceptual representations
        R = self.perceptual_encoder(images) # (num_singletons, num_patches, embed_dim)
        
        # 3. Define the formal concepts dynamically
        concept_defs = {}
        
        # Universal concept
        concept_defs["C0 (Universal)"] = list(range(num_singletons))
        
        concept_idx = 1
        # Color concepts
        for color in colors:
            concept_defs[f"C{concept_idx} ({color.capitalize()})"] = [
                i for i, (c, s) in enumerate(combinations) if c == color
            ]
            concept_idx += 1

        # Shape concepts
        for shape in shapes:
            concept_defs[f"C{concept_idx} ({shape.capitalize()})"] = [
                i for i, (c, s) in enumerate(combinations) if s == shape
            ]
            concept_idx += 1
            
        # Singleton concepts
        for i, (c, s) in enumerate(combinations):
            concept_defs[f"C{concept_idx} ({c.capitalize()} {s.capitalize()})"] = [i]
            concept_idx += 1
            
        concept_labels = list(concept_defs.keys())
        num_concepts = len(concept_labels)
        
        # Extract projectors for all concepts
        P_attr_concepts = []
        P_obj_concepts = []
        X_attr_concepts = {}
        X_obj_concepts = {}
        
        for name, indices in concept_defs.items():
            subset = R[indices].unsqueeze(0) # (1, subset_size, num_patches, embed_dim)
            X_attr, X_obj, P_attr, P_obj = self.concept_encoder(subset)
            P_attr_concepts.append(P_attr.squeeze(0)) # (ambient_dim, ambient_dim)
            P_obj_concepts.append(P_obj.squeeze(0))   # (ambient_dim, ambient_dim)
            X_attr_concepts[name] = X_attr.squeeze(0) # (n_attr, ambient_dim)
            X_obj_concepts[name] = X_obj.squeeze(0)   # (n_obj, ambient_dim)
            
        P_attr_concepts = torch.stack(P_attr_concepts) # (num_concepts, D, D)
        P_obj_concepts = torch.stack(P_obj_concepts)   # (num_concepts, D, D)
        X_attr_concepts_tensor = torch.stack(list(X_attr_concepts.values())) # (num_concepts, n_attr, D)
        
        # Extract projectors for the singletons (last `num_singletons` concepts)
        P_attr_singletons = P_attr_concepts[-num_singletons:] # (num_singletons, D, D)
        P_obj_singletons = P_obj_concepts[-num_singletons:]   # (num_singletons, D, D)
        X_attr_singletons = X_attr_concepts_tensor[-num_singletons:] # (num_singletons, n_attr, D)
        
        # 4. Compute Inclusion Matrices
        obj_inclusion = np.zeros((num_singletons, num_concepts))
        attr_inclusion = np.zeros((num_singletons, num_concepts))
        
        for i in range(num_singletons):
            for j in range(num_concepts):
                # Object: Is singleton 'i' included in concept 'j'?
                inc_obj = get_inclusion(P_obj_singletons[i].unsqueeze(0), P_obj_concepts[j].unsqueeze(0))
                obj_inclusion[i, j] = inc_obj.item()
                
                # Attribute: Reverse inclusion due to Galois connection! 
                # Is concept 'j's intent included in singleton 'i's intent?
                if getattr(self, "binary_inclusion", False):
                    inc_attr = get_binary_inclusion(
                        X_attr_concepts_tensor[j].unsqueeze(0), 
                        X_attr_singletons[i].unsqueeze(0)
                    )
                else:
                    inc_attr = get_inclusion(P_attr_concepts[j].unsqueeze(0), P_attr_singletons[i].unsqueeze(0))
                attr_inclusion[i, j] = inc_attr.item()

        # 5. Plotting
        # Make figure wider based on number of concepts
        fig, axes = plt.subplots(2, 1, figsize=(max(12, num_concepts * 0.8), max(10, num_singletons * 1.5)))
        
        sns.heatmap(obj_inclusion, annot=True, cmap="Blues", vmin=0, vmax=1, 
                    xticklabels=concept_labels, yticklabels=singleton_labels, ax=axes[0])
        axes[0].set_title("Object Subspace Inclusion: $P(S_{singleton}^{obj} \subseteq S_{concept}^{obj})$\\n(Expect ~1 for valid combinations)")
        axes[0].tick_params(axis='y', labelrotation=0)
        axes[0].tick_params(axis='x', labelrotation=45)
        
        sns.heatmap(attr_inclusion, annot=True, cmap="Oranges", vmin=0, vmax=1, 
                    xticklabels=concept_labels, yticklabels=singleton_labels, ax=axes[1])
        axes[1].set_title("Attribute Subspace Inclusion: $P(S_{concept}^{attr} \subseteq S_{singleton}^{attr})$\\n(Galois Reverse: Expect ~1 for valid combinations)")
        axes[1].tick_params(axis='y', labelrotation=0)
        axes[1].tick_params(axis='x', labelrotation=45)
        
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # Convert buffer to a numpy array, then to a torch Tensor (C, H, W)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img_arr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        # OpenCV uses BGR, convert to RGB
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        
        # Convert to standard PyTorch image format (C, H, W)
        heatmap_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        # 6. Plotting Basis Vectors
        if len(colors) > 0 and len(shapes) > 0:
            c0 = colors[0].capitalize()
            s0 = shapes[0].capitalize()
            s1 = shapes[1].capitalize() if len(shapes) > 1 else s0
            targets = ["(Universal)", f"({c0})", f"({c0} {s0})", f"({c0} {s1})"]
        else:
            targets = ["(Universal)"]

        keys_to_plot = []
        for target in targets:
            for k in concept_labels:
                if target in k:
                    if k not in keys_to_plot:
                        keys_to_plot.append(k)
                    break

        if len(keys_to_plot) > 0:
            fig, axes = plt.subplots(len(keys_to_plot), 2, figsize=(16, 4 * len(keys_to_plot)))
            if len(keys_to_plot) == 1:
                axes = np.expand_dims(axes, axis=0)
            
            for i, key in enumerate(keys_to_plot):
                x_attr = X_attr_concepts[key].cpu().numpy()
                x_obj = X_obj_concepts[key].cpu().numpy()
                
                if x_attr.ndim == 1:
                    x_attr = x_attr[np.newaxis, :]
                if x_obj.ndim == 1:
                    x_obj = x_obj[np.newaxis, :]
                    
                ax_attr = axes[i, 0]
                ax_obj = axes[i, 1]
                
                n_attr, d_ambient_attr = x_attr.shape
                bar_width_attr = 0.8 / n_attr
                for r in range(n_attr):
                    ax_attr.bar(np.arange(d_ambient_attr) + r * bar_width_attr, x_attr[r], 
                                width=bar_width_attr, alpha=0.7, label=f"Basis {r}")
                ax_attr.set_title(f"X_attr: {key}")
                ax_attr.set_xlabel("Dimension Index")
                ax_attr.set_ylabel("Value")
                if n_attr > 1:
                    ax_attr.legend()
                    
                n_obj, d_ambient_obj = x_obj.shape
                bar_width_obj = 0.8 / n_obj
                for r in range(n_obj):
                    ax_obj.bar(np.arange(d_ambient_obj) + r * bar_width_obj, x_obj[r], 
                               width=bar_width_obj, alpha=0.7, label=f"Basis {r}")
                ax_obj.set_title(f"X_obj: {key}")
                ax_obj.set_xlabel("Dimension Index")
                ax_obj.set_ylabel("Value")
                if n_obj > 1:
                    ax_obj.legend()
                    
            plt.tight_layout()
            buf_bases = io.BytesIO()
            plt.savefig(buf_bases, format='png', bbox_inches='tight')
            plt.close(fig)
            buf_bases.seek(0)
            
            img_arr_bases = np.frombuffer(buf_bases.getvalue(), dtype=np.uint8)
            img_arr_bases = cv2.imdecode(img_arr_bases, cv2.IMREAD_COLOR)
            img_arr_bases = cv2.cvtColor(img_arr_bases, cv2.COLOR_BGR2RGB)
            basis_tensor = torch.from_numpy(img_arr_bases).permute(2, 0, 1)
        else:
            basis_tensor = None
        
        self.train() # restore training mode
        return heatmap_tensor, basis_tensor

    def configure_optimizers(self):

        if self.config["trainer"]["optimizer"]["type"] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config["trainer"]["optimizer"]["config"]["lr"],
            )
            return optimizer
        else:
            raise ValueError("Optimizer not found")

    def on_train_epoch_end(self):
        if self.viz_datapoint is not None:
            fig = self.create_reconstruction_figure(
                self.viz_datapoint, 
                self.current_epoch
            )
            self.logger.experiment.log({"reconstruction_visualization": fig})
            
            self.viz_datapoint = None
            plt.close(fig)

        inclusion_fig, basis_fig = self.log_concept_lattice_inclusion()
        log_dict = {
            "lattice_inclusion_heatmap": wandb.Image(inclusion_fig),
        }
        if basis_fig is not None:
            log_dict["concept_basis_vectors"] = wandb.Image(basis_fig)
            
        self.logger.experiment.log(log_dict)

    @staticmethod
    def create_reconstruction_figure(data, epoch):
        original_imgs = data["original_images"]
        recon_imgs = data["reconstructed_images"]

        import numpy as np
        grid_in = torchvision.utils.make_grid(original_imgs, nrow=original_imgs.shape[0], padding=2)
        grid_in = grid_in.permute(1, 2, 0).numpy()

        grid_recon = torchvision.utils.make_grid(recon_imgs, nrow=recon_imgs.shape[0], padding=2)
        grid_recon = grid_recon.permute(1, 2, 0).numpy()

        fig = plt.figure(figsize=(12, 6))
        plt.suptitle(f"Epoch {epoch} Reconstructions", fontsize=16)

        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(np.clip(grid_in, 0, 1))
        ax1.set_title("Original Images")
        ax1.axis("off")

        ax2 = plt.subplot(2, 1, 2)
        ax2.imshow(np.clip(grid_recon, 0, 1))
        ax2.set_title("Reconstructed Images")
        ax2.axis("off")

        plt.tight_layout()
        
        return fig