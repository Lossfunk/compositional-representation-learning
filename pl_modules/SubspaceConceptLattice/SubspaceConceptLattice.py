import itertools
import random
import io

import numpy as np
import cv2
import torch
import torch.nn as nn
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
from .concept_utils import get_inclusion, get_residual_inclusion, ridge_projector, get_binary_inclusion

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
        self.pairwise_level_by_level = self.model_config.get("pairwise_level_by_level", False)
        self.num_levels = self.model_config.get("num_levels", 3)
        self.max_pairs_per_level = self.model_config.get("max_pairs_per_level", 10)
        _ic_uc_mode = self.model_config.get("ic_uc_mode", "mse")  # fallback for both
        self.ic_mode = self.model_config.get("ic_mode", _ic_uc_mode)  # "mse" or "bounds"
        self.uc_mode = self.model_config.get("uc_mode", _ic_uc_mode)  # "mse" or "bounds"

        self.inclusion_type = self.model_config.get("inclusion_type", "trace_ratio")
        self.residual_inclusion_gamma = float(self.model_config.get("residual_inclusion_gamma", 1.0))

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

        if self.enable_intermediate_representations or self.pairwise_level_by_level:
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

        # Learnable orthogonal matrix W mapping attr subspace into obj coordinates.
        # W P_attr W^T lives in the same basis as P_obj, enabling direct comparison.
        self.attr_obj_orthogonality_direct = self.model_config.get("attr_obj_orthogonality_direct", False)
        if self.loss_weights.get("attr_obj_orthogonality_loss", 0) > 0 and not self.attr_obj_orthogonality_direct:
            W_linear = nn.Linear(self.ambient_dim, self.ambient_dim, bias=False)
            self.W = torch.nn.utils.parametrizations.orthogonal(W_linear)
        else:
            self.W = None

        self.viz_datapoint = None

    def forward(self, x):
        images = x["images"] # (B, 3, H, W)
        B = images.shape[0]
        representations = self.perceptual_encoder(images) # (B, num_patches, embed_dim)

        if self.pairwise_level_by_level:
            return self._forward_pairwise(images, representations, B)

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

    def _forward_pairwise(self, images, representations, B):
        """
        Build concepts level-by-level via pairwise representation combination.

        Level 0: singleton representations from the perceptual encoder.
        Level L (1..num_levels): randomly sample max_pairs_per_level pairs
                 from level L-1, combine each pair via RepresentationCombiner
                 to produce new (1, P, E) representations.

        After all levels are built, pass ALL representations (from all levels)
        through the concept encoder once to get subspaces.
        """
        P, E = representations.shape[1], representations.shape[2]

        # ── Level 0: Singleton representations ────────────────────────
        # Each singleton is just its perceptual representation: (B, P, E)
        all_reprs = [representations]  # list of (N_level, P, E)
        all_levels = [torch.zeros(B, device=self.device, dtype=torch.long)]

        # Parent tracking: (child_global_idx, parent_a_global_idx, parent_b_global_idx)
        parent_pairs = []
        offset = B

        # prev_level_reprs: (N_prev, P, E) — representations from the previous level
        # prev_level_offset: global index of the first entry in prev_level
        prev_level_reprs = representations  # (B, P, E)
        prev_level_offset = 0

        for level in range(1, self.num_levels + 1):
            N_prev = prev_level_reprs.shape[0]
            if N_prev < 2:
                break

            # All possible pairs from previous level
            all_pair_indices = list(itertools.combinations(range(N_prev), 2))
            # Randomly sample up to max_pairs_per_level
            if len(all_pair_indices) > self.max_pairs_per_level:
                all_pair_indices = random.sample(all_pair_indices, self.max_pairs_per_level)

            N_pairs = len(all_pair_indices)
            idx_a = [p[0] for p in all_pair_indices]
            idx_b = [p[1] for p in all_pair_indices]

            # Stack pairs: (N_pairs, 2, P, E)
            R_pairs = torch.stack([
                prev_level_reprs[idx_a],
                prev_level_reprs[idx_b],
            ], dim=1)

            # Combine each pair into a single representation: (N_pairs, 1, P, E)
            R_combined = self.representation_combiner(R_pairs)
            R_combined = R_combined.squeeze(1)  # (N_pairs, P, E)

            all_reprs.append(R_combined)
            all_levels.append(
                torch.full((N_pairs,), level, device=self.device, dtype=torch.long)
            )

            # Track parent pairs (global indices)
            for i, (a, b) in enumerate(all_pair_indices):
                parent_pairs.append((offset + i, prev_level_offset + a, prev_level_offset + b))

            prev_level_offset = offset
            offset += N_pairs
            prev_level_reprs = R_combined

        # ── Concatenate all representations ───────────────────────────
        all_reprs_tensor = torch.cat(all_reprs, dim=0)  # (total, P, E)
        levels_tensor = torch.cat(all_levels, dim=0)
        singletons_mask = (levels_tensor == 0)

        # ── Single concept encoder call on all representations ────────
        # Concept encoder expects (N, S, P, E) where S=1 for combined reprs
        R_all = all_reprs_tensor.unsqueeze(1)  # (total, 1, P, E)
        X_attr, X_obj, P_attr, P_obj = self.concept_encoder(
            R_all, current_epoch=getattr(self, 'current_epoch', None)
        )

        # ── Reconstruct from singletons only ──────────────────────────
        X_attr_singletons = X_attr[:B]
        X_attr_singletons_dec = self.concept_encoder.attr_dec(X_attr_singletons)
        reconstructed_images = self.decoder(X_attr_singletons_dec)

        return {
            "images": images,
            "reconstructed_images": reconstructed_images,
            "X_attr_tensor": X_attr,
            "X_obj_tensor": X_obj,
            "P_attr_tensor": P_attr,
            "P_obj_tensor": P_obj,
            "levels_tensor": levels_tensor,
            "singletons_mask": singletons_mask,
            "parent_pairs": parent_pairs,
        }

    def _compute_inclusion(self, P_sub, P_super):
        """Dispatches to trace-ratio or residual inclusion based on config."""
        if self.inclusion_type == "residual":
            return get_residual_inclusion(P_sub, P_super, gamma=self.residual_inclusion_gamma)
        return get_inclusion(P_sub, P_super)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        images = outputs["images"] # (B, 3, H, W)
        reconstructed_images = outputs["reconstructed_images"] # (B, 3, H, W)
        B = images.shape[0]

        X_attr_tensor = outputs["X_attr_tensor"] # (B_total, n_attr, ambient_dim)
        X_obj_tensor = outputs["X_obj_tensor"] # (B_total, n_obj, ambient_dim)
        P_attr_tensor = outputs["P_attr_tensor"] # (B_total, ambient_dim, ambient_dim)
        P_obj_tensor = outputs["P_obj_tensor"] # (B_total, ambient_dim, ambient_dim)
        singletons_mask = outputs["singletons_mask"]
        B_total = X_attr_tensor.shape[0]

        # These are only present in the old combinatorial path
        cardinalities_tensor = outputs.get("cardinalities_tensor")
        combination_indices = outputs.get("combination_indices")

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

        obj_ranks = torch.einsum("bii->b", P_obj_tensor)
        attr_ranks = torch.einsum("bii->b", P_attr_tensor)

        if self.pairwise_level_by_level:
            # ── Pairwise mode: IC/UC bounds from parent pairs ─────────
            # All cardinality-based losses are disabled.
            union_consistency_loss = 0
            intersection_consistency_loss = 0
            modular_subspace_loss = 0
            galois_attr_loss = 0
            galois_obj_loss = 0
            attr_sink_loss = 0
            loss_obj_card_prop = 0
            loss_attr_card_inv_prop = 0
            loss_attr_obj_inv_prop = 0

            parent_pairs = outputs.get("parent_pairs", [])
            if parent_pairs:
                pp_child = torch.tensor([p[0] for p in parent_pairs], device=self.device)
                pp_a = torch.tensor([p[1] for p in parent_pairs], device=self.device)
                pp_b = torch.tensor([p[2] for p in parent_pairs], device=self.device)

                # IC: P_attr_child vs intersect(P_attr_parent_a, P_attr_parent_b)
                # Pairwise intersection: 0.5 * (P_a @ P_b + P_b @ P_a)
                P_attr_c = P_attr_tensor[pp_child]
                P_attr_pa = P_attr_tensor[pp_a]
                P_attr_pb = P_attr_tensor[pp_b]
                P_attr_inter = 0.5 * (torch.bmm(P_attr_pa, P_attr_pb) + torch.bmm(P_attr_pb, P_attr_pa))

                if self.ic_mode == "bounds":
                    ic_incl = self._compute_inclusion(P_attr_c, P_attr_inter)
                    ic_incl = torch.clamp(ic_incl, 1e-6, 1.0 - 1e-6)
                    intersection_consistency_loss = F.binary_cross_entropy(
                        ic_incl, torch.ones_like(ic_incl)
                    )
                else:
                    intersection_consistency_loss = F.mse_loss(P_attr_c, P_attr_inter)

                # UC: union(P_obj_parent_a, P_obj_parent_b) vs P_obj_child
                P_obj_c = P_obj_tensor[pp_child]
                X_obj_pa = X_obj_tensor[pp_a]
                X_obj_pb = X_obj_tensor[pp_b]
                X_obj_union_basis = torch.cat([X_obj_pa, X_obj_pb], dim=1)  # (N, 2*n_obj, D)
                P_obj_union = ridge_projector(X_obj_union_basis, lbd=self.lbd)

                if self.uc_mode == "bounds":
                    uc_incl = get_inclusion(P_obj_union, P_obj_c)
                    uc_incl = torch.clamp(uc_incl, 1e-6, 1.0 - 1e-6)
                    union_consistency_loss = F.binary_cross_entropy(
                        uc_incl, torch.ones_like(uc_incl)
                    )
                else:
                    union_consistency_loss = F.mse_loss(P_obj_c, P_obj_union)

        else:
            # ── Combinatorial mode: cardinality-based losses ──────────
            total_union_consistency_loss = 0
            total_intersection_consistency_loss = 0
            total_modular_subspace_loss = 0
            total_galois_attr_loss = 0
            total_galois_obj_loss = 0
            total_comparisons = 0
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
                if self.uc_mode == "bounds":
                    uc_incl = get_inclusion(P_obj_union, cardinality_P_obj)
                    uc_incl = torch.clamp(uc_incl, 1e-6, 1.0 - 1e-6)
                    total_union_consistency_loss += F.binary_cross_entropy(uc_incl, torch.ones_like(uc_incl), reduction="sum")
                else:
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
                        if self.ic_mode == "bounds":
                            ic_incl = self._compute_inclusion(cardinality_P_attr, combination_singletons_P_attr_inter)
                            ic_incl = torch.clamp(ic_incl, 1e-6, 1.0 - 1e-6)
                            total_intersection_consistency_loss += F.binary_cross_entropy(ic_incl, torch.ones_like(ic_incl), reduction="sum")
                        else:
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
                    if self.ic_mode == "bounds":
                        ic_incl = self._compute_inclusion(cardinality_P_attr, combination_singletons_P_attr_inter)
                        ic_incl = torch.clamp(ic_incl, 1e-6, 1.0 - 1e-6)
                        total_intersection_consistency_loss += F.binary_cross_entropy(ic_incl, torch.ones_like(ic_incl), reduction="sum")
                    else:
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
                    galois_attr_inclusion = self._compute_inclusion(P_sub=P_attr_c_flat, P_super=P_attr_s_flat) # (num_combinations * k,)

                galois_obj_inclusion = self._compute_inclusion(P_sub=P_obj_s_flat, P_super=P_obj_c_flat) # (num_combinations * k,)

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

            ## Proportionality Losses
            cardinalities = cardinalities_tensor.float()
            loss_obj_card_prop = self.proportionality_loss(cardinalities, obj_ranks, inverse=False)
            loss_attr_card_inv_prop = self.proportionality_loss(cardinalities, attr_ranks, inverse=True)
            loss_attr_obj_inv_prop = self.proportionality_loss(attr_ranks, obj_ranks, inverse=True)

        if self.global_galois_loss_start_epoch is not None and self.current_epoch >= self.global_galois_loss_start_epoch:
            global_galois_loss = self.compute_global_galois_loss(P_attr_tensor, P_obj_tensor, X_attr_tensor)
        else:
            global_galois_loss = 0

        basis_orthogonality_loss = self.compute_basis_orthogonality_loss(X_attr_singletons)
        basis_sparsity_loss = self.compute_basis_sparsity_loss(X_attr_singletons)

        ## Repulsion Losses
        repulsion_loss_obj = self.compute_soft_repulsion(P_obj_singletons)
        repulsion_loss_attr = self.compute_soft_repulsion(P_attr_singletons)

        ## Residual Orthogonality and Rank Conservation Losses
        residual_orthogonality_loss, rank_conservation_loss = self.compute_residual_orthogonality_loss(
            P_attr_singletons, X_attr_singletons
        )

        ## Singleton Obj Subspace Orthogonality Loss
        obj_subspace_orthogonality_loss = self.compute_obj_subspace_orthogonality_loss(
            P_obj_singletons, P_attr_singletons=P_attr_singletons
        )

        ## Attr-Obj Orthogonality Loss (via learnable W)
        attr_obj_orthogonality_loss = self.compute_attr_obj_orthogonality_loss(
            P_attr_tensor, P_obj_tensor
        )

        ## Cross-Slot Decorrelation Loss
        cross_slot_decorrelation_loss = self.compute_cross_slot_decorrelation_loss(
            X_attr_singletons
        )

        ## Galois Cycle Loss
        galois_cycle_loss = self.compute_galois_cycle_loss(
            P_attr_tensor, P_obj_tensor,
            X_attr_tensor, X_obj_tensor,
            P_attr_singletons, P_obj_singletons,
            X_attr_singletons, X_obj_singletons,
        )

        ## Rank Complementarity Loss
        rank_complementarity_loss = self.compute_rank_complementarity_loss(
            P_attr_tensor, P_obj_tensor, singletons_mask
        )

        ## Margin-Based Topological Sort Loss
        margin_topological_sort_loss = self.compute_margin_topological_sort_loss(
            P_attr_tensor,
            P_attr_singletons=P_attr_singletons,
            cardinalities_tensor=cardinalities_tensor,
            combination_indices=combination_indices,
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
            self.loss_weights.get("basis_sparsity_loss", 0) * basis_sparsity_loss +
            self.loss_weights.get("attr_obj_orthogonality_loss", 0) * attr_obj_orthogonality_loss +
            self.loss_weights.get("obj_subspace_orthogonality_loss", 0) * obj_subspace_orthogonality_loss +
            self.loss_weights.get("cross_slot_decorrelation_loss", 0) * cross_slot_decorrelation_loss +
            self.loss_weights.get("galois_cycle_loss", 0) * galois_cycle_loss +
            self.loss_weights.get("rank_complementarity_loss", 0) * rank_complementarity_loss +
            self.loss_weights.get("margin_topological_sort_loss", 0) * margin_topological_sort_loss
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
            "attr_obj_orthogonality_loss": attr_obj_orthogonality_loss,
            "obj_subspace_orthogonality_loss": obj_subspace_orthogonality_loss,
            "cross_slot_decorrelation_loss": cross_slot_decorrelation_loss,
            "galois_cycle_loss": galois_cycle_loss,
            "rank_complementarity_loss": rank_complementarity_loss,
            "margin_topological_sort_loss": margin_topological_sort_loss,
            "total_loss": total_loss
        }

        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)

        if not self.pairwise_level_by_level:
            for cardinality in range(1, B + 1):
                mask = (cardinalities_tensor == cardinality)
                if mask.any():
                    avg_attr_rank = attr_ranks[mask].mean()
                    avg_obj_rank = obj_ranks[mask].mean()
                    self.log(f"rank/attr_cardinality_{cardinality}", avg_attr_rank, on_epoch=True, prog_bar=False)
                    self.log(f"rank/obj_cardinality_{cardinality}", avg_obj_rank, on_epoch=True, prog_bar=False)
        else:
            levels_tensor = outputs["levels_tensor"]
            for level in range(self.num_levels + 1):
                mask = (levels_tensor == level)
                if mask.any():
                    avg_attr_rank = attr_ranks[mask].mean()
                    avg_obj_rank = obj_ranks[mask].mean()
                    self.log(f"rank/attr_level_{level}", avg_attr_rank, on_epoch=True, prog_bar=False)
                    self.log(f"rank/obj_level_{level}", avg_obj_rank, on_epoch=True, prog_bar=False)

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
        elif self.inclusion_type == "residual":
            # Residual inclusion: exp(-gamma * ReLU(Tr(P_i) - Tr(P_i P_j)))
            P_attr_flat = P_attr_tensor.reshape(N, -1)
            overlap_attr = torch.matmul(P_attr_flat, P_attr_flat.t())
            trace_attr = torch.einsum("bii->b", P_attr_tensor)
            residual_attr = F.relu(trace_attr.unsqueeze(1) - overlap_attr)
            inc_attr = torch.exp(-self.residual_inclusion_gamma * residual_attr)
        else:
            # Trace-ratio inclusion (original)
            P_attr_flat = P_attr_tensor.reshape(N, -1)
            overlap_attr = torch.matmul(P_attr_flat, P_attr_flat.t())
            trace_attr = torch.einsum("bii->b", P_attr_tensor).clamp(min=1e-6)
            inc_attr = overlap_attr / trace_attr.unsqueeze(1) # (N, N)

        if self.inclusion_type == "residual":
            P_obj_flat = P_obj_tensor.reshape(N, -1)
            overlap_obj = torch.matmul(P_obj_flat, P_obj_flat.t())
            trace_obj = torch.einsum("bii->b", P_obj_tensor)
            residual_obj = F.relu(trace_obj.unsqueeze(1) - overlap_obj)
            inc_obj = torch.exp(-self.residual_inclusion_gamma * residual_obj)
        else:
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

    def compute_attr_obj_orthogonality_loss(self, P_attr, P_obj):
        """
        Forces attr and obj subspaces to encode complementary information.

        Two modes controlled by `attr_obj_orthogonality_direct`:
        - False (default): Uses a learned orthogonal W to map attr into obj
          coordinates before measuring overlap: Tr(P_obj @ W P_attr W^T) → 0.
        - True: Directly measures overlap in the same space: Tr(P_obj @ P_attr) → 0.

        Args:
            P_attr: (B_total, D, D) — attr projectors for ALL concepts
            P_obj:  (B_total, D, D) — obj projectors for ALL concepts
        Returns:
            Scalar loss
        """
        if self.attr_obj_orthogonality_direct:
            overlap = torch.einsum("bij,bji->b", P_obj, P_attr)
            return overlap.mean()

        if self.W is None:
            return torch.tensor(0.0, device=P_attr.device)

        W_mat = self.W.weight  # (D, D), orthogonal
        # Transform attr projectors into obj coordinate system
        # P_attr_trans = W @ P_attr @ W^T, batched
        P_attr_trans = torch.einsum("ij,bjk,lk->bil", W_mat, P_attr, W_mat)
        # Overlap = Tr(P_obj @ P_attr_trans) per concept
        overlap = torch.einsum("bij,bji->b", P_obj, P_attr_trans)
        return overlap.mean()

    def compute_obj_subspace_orthogonality_loss(self, P_obj_singletons, P_attr_singletons=None):
        """
        Ensures singleton object subspaces are structured: collectively span n_obj
        distinct directions, with pairwise overlap polarized toward {0, 1}.

        Uses covariance-ridge rank to push collective rank toward n_obj.
        For pairwise structure, uses projector overlap polarization instead of
        pushing all pairs to zero overlap (which is wrong when objects share features).

        obj_subspace_orthogonality_mode config controls pairwise behavior:
          "ortho" (default/legacy): push all pairwise overlap to 0
          "polarize": push pairwise inclusion toward {0, 1}
          "bimodal": min(x², (1-x)²) per pair
          "attr_guided": uses attr subspace overlap to dynamically determine
              which singletons are duplicates (same attributes → align obj)
              vs distinct (different attributes → orthogonalize obj).
              Requires P_attr_singletons argument.
        """
        B = P_obj_singletons.shape[0]
        D = P_obj_singletons.shape[1]
        if B < 2:
            return torch.tensor(0.0, device=P_obj_singletons.device)

        obj_ortho_mode = self.model_config.get("obj_subspace_orthogonality_mode", "ortho")
        obj_balance_weight = float(self.model_config.get("obj_ortho_balance_weight", 1.0))

        # Vectorize projectors: (B, D*D)
        P_vec = P_obj_singletons.reshape(B, -1)
        # Normalize each vectorized projector
        P_norm = F.normalize(P_vec, p=2, dim=-1)  # (B, D*D)

        # Covariance-based rank: singleton obj projectors should collectively span
        # n_obj distinct directions (one per unique object in the dataset).
        C = torch.mm(P_norm.t(), P_norm)  # (D*D, D*D)
        C_ridge = C + self.lbd * torch.eye(D * D, device=self.device)
        P_cov = torch.mm(C, torch.linalg.inv(C_ridge))
        current_rank = torch.trace(P_cov)
        target_rank = float(self.n_obj)
        loss_rank = F.mse_loss(current_rank, torch.tensor(target_rank, device=self.device))

        if obj_ortho_mode == "attr_guided":
            assert P_attr_singletons is not None, \
                "attr_guided mode requires P_attr_singletons"
            # Obj pairwise overlap (normalized Frobenius)
            G_obj = torch.mm(P_norm, P_norm.t())  # (B, B)

            # Discover duplicates from the attr subspace (detached — no gradient
            # feedback into attr space, so the obj loss can't collapse attributes).
            with torch.no_grad():
                overlap_attr = torch.einsum(
                    "bij,kji->bk", P_attr_singletons, P_attr_singletons
                )  # (B, B)
                # Two singletons are the same object iff they share ALL attributes.
                # With n_attr=2, threshold at 1.5 cleanly separates
                # same-object (overlap≈2) from different-object (overlap≈0 or ≈1).
                duplicate_mask = (overlap_attr > (self.n_attr - 0.5)).float()

            loss_pairwise = F.mse_loss(G_obj, duplicate_mask)

        elif obj_ortho_mode in ("polarize", "bimodal"):
            # Projector overlap: Tr(P_i @ P_j)/Tr(P_i) = inclusion(P_i ⊆ P_j)
            traces = torch.einsum("bii->b", P_obj_singletons)  # (B,)
            overlap = torch.einsum("bij,kji->bk", P_obj_singletons, P_obj_singletons)  # (B, B)
            inclusion = overlap / (traces.unsqueeze(1) + 1e-8)  # (B, B)
            mask = ~torch.eye(B, dtype=torch.bool, device=P_obj_singletons.device)
            incl_vals = inclusion[mask].clamp(0.0, 1.0)

            if obj_ortho_mode == "polarize":
                # x*(1-x): smooth, zero at {0,1}, max at 0.5
                loss_pairwise = torch.mean(incl_vals * (1 - incl_vals))
            else:
                # bimodal: min(x², (1-x)²): cusp at 0.5, steeper near targets
                loss_pairwise = torch.mean(torch.min(incl_vals**2, (1 - incl_vals)**2))

            # Balance: pull mean inclusion toward p_target = 1/n_obj
            # (each singleton is unique → same-type fraction ≈ 1/n_obj in a random batch)
            p = incl_vals.mean().clamp(1e-6, 1 - 1e-6)
            p_target = max(1.0 / self.n_obj, 1e-6)
            p_target = min(p_target, 1.0 - 1e-6)
            loss_balance_obj = (p_target * torch.log(torch.tensor(p_target, device=P_obj_singletons.device) / p)
                                + (1 - p_target) * torch.log(torch.tensor(1 - p_target, device=P_obj_singletons.device) / (1 - p)))
            loss_pairwise = loss_pairwise + loss_balance_obj * obj_balance_weight
        else:
            # Legacy "ortho": push all pairwise overlap to zero
            G = torch.mm(P_norm, P_norm.t())  # (B, B)
            mask = ~torch.eye(B, dtype=torch.bool, device=self.device)
            loss_pairwise = torch.mean(G[mask] ** 2)

        return loss_rank + loss_pairwise

    def compute_cross_slot_decorrelation_loss(self, X):
        """
        Forces attr slots to encode DIFFERENT generative factors by penalizing
        pairs where both slots agree on high absolute cosine similarity.

        For each slot, builds the B×B absolute cosine similarity Gram matrix S_s.
        Then penalizes the Frobenius inner product: sum(|S_0[i,j]| * |S_1[i,j]|).

        If slot 0 has cos=1 for same-color pair (i,j), this forces slot 1 to have
        cos≈0 for that pair. The only way to satisfy both this loss and bimodal
        basis_orthogonality is for each slot to encode a different factor.

        Args:
            X: (B, n_attr, ambient_dim) — singleton attr basis vectors
        Returns:
            Scalar loss
        """
        B, N, D = X.shape
        if B < 2 or N < 2:
            return torch.tensor(0.0, device=X.device)

        # Build per-slot absolute cosine similarity Gram matrices
        gram_matrices = []
        for slot_idx in range(N):
            vecs = X[:, slot_idx, :]  # (B, D)
            vecs_norm = F.normalize(vecs, p=2, dim=-1)  # (B, D)
            S = torch.mm(vecs_norm, vecs_norm.t())  # (B, B)
            gram_matrices.append(S.abs())

        # Frobenius inner product of absolute Gram matrices across slot pairs
        loss = torch.tensor(0.0, device=X.device)
        n_pairs = 0
        for i in range(N):
            for j in range(i + 1, N):
                # Exclude diagonal (self-similarity is always 1)
                mask = ~torch.eye(B, dtype=torch.bool, device=X.device)
                product = gram_matrices[i][mask] * gram_matrices[j][mask]
                loss = loss + product.mean()
                n_pairs += 1

        return loss / max(1, n_pairs)

    def _sharpen_weights(self, raw_weights, mode, threshold, tau):
        """
        Sharpens soft inclusion weights into near-binary gates.

        Args:
            raw_weights: (*, B) raw inclusion scores in [0, 1]
            mode: "temperature" (W3) or "ste" (W4)
            threshold: center point for the gate
            tau: temperature (lower = sharper). Only used in "temperature" mode.
        Returns:
            Sharpened weights, same shape as raw_weights
        """
        if mode == "temperature":
            return torch.sigmoid((raw_weights - threshold) / tau)
        elif mode == "ste":
            hard = (raw_weights > threshold).float()
            # Straight-through estimator: hard forward, soft backward
            return hard + raw_weights - raw_weights.detach()
        else:
            return raw_weights

    def compute_galois_cycle_loss(
        self,
        P_attr_all, P_obj_all,
        X_attr_all, X_obj_all,
        P_attr_singletons, P_obj_singletons,
        X_attr_singletons, X_obj_singletons,
    ):
        """
        Galois Cycle Loss: enforces the FCA derivation operators as a
        parameter-free bijection between attr and obj subspaces.

        β (attr→obj): For each concept, query which singletons in the batch
        contain its attributes (via inclusion), then build the obj target as
        the ridge projector of the weighted union of those singletons' obj
        basis vectors.

        α (obj→attr): For each concept, query which singletons' obj subspaces
        are contained in its obj subspace, then build the attr target as the
        power-iteration intersection of those singletons' attr projectors.

        Loss = MSE(P_obj_actual, β(P_attr)) + MSE(P_attr_actual, α(P_obj))

        Config keys (under model.config):
            galois_cycle_weight_mode: "temperature" or "ste"
            galois_cycle_threshold: gate center (default 0.8)
            galois_cycle_tau: temperature for sigmoid sharpening (default 0.1)

        Args:
            P_attr_all:        (B_total, D, D) attr projectors for all concepts
            P_obj_all:         (B_total, D, D) obj projectors for all concepts
            X_attr_all:        (B_total, n_attr, D) attr basis vectors for all concepts
            X_obj_all:         (B_total, n_obj, D) obj basis vectors for all concepts
            P_attr_singletons: (B, D, D) attr projectors for singletons
            P_obj_singletons:  (B, D, D) obj projectors for singletons
            X_attr_singletons: (B, n_attr, D) attr basis vectors for singletons
            X_obj_singletons:  (B, n_obj, D) obj basis vectors for singletons
        Returns:
            Scalar loss (loss_beta + loss_alpha)
        """
        B_total = P_attr_all.shape[0]
        B = P_attr_singletons.shape[0]
        D = P_attr_all.shape[1]

        if B < 2:
            return torch.tensor(0.0, device=P_attr_all.device)

        model_cfg = self.config["model"]["config"]
        weight_mode = model_cfg.get("galois_cycle_weight_mode", "temperature")
        threshold = model_cfg.get("galois_cycle_threshold", 0.8)
        tau = model_cfg.get("galois_cycle_tau", 0.1)

        # ──────────────────────────────────────────────────────────
        # β direction: attr → obj
        # "Given this concept's attributes, which singletons have them?"
        # w_j = Inclusion(P_attr_concept ⊆ P_attr_singleton_j)
        #     = Tr(P_attr_singleton_j @ P_attr_concept) / Tr(P_attr_concept)
        # ──────────────────────────────────────────────────────────

        # Tr(P_s[j] @ P_c[i]) for all (i, j) pairs
        # P_attr_all: (B_total, D, D), P_attr_singletons: (B, D, D)
        overlap_beta = torch.einsum(
            "aij,bji->ab", P_attr_all, P_attr_singletons
        )  # (B_total, B)
        if self.inclusion_type == "residual":
            trace_query_attr = torch.einsum("aii->a", P_attr_all).unsqueeze(1)  # (B_total, 1)
            residual_beta = F.relu(trace_query_attr - overlap_beta)
            raw_incl_beta = torch.clamp(
                torch.exp(-self.residual_inclusion_gamma * residual_beta), 0.0, 1.0
            )  # (B_total, B)
        else:
            rank_query_attr = (
                torch.einsum("aii->a", P_attr_all).unsqueeze(1) + 1e-6
            )  # (B_total, 1)
            raw_incl_beta = torch.clamp(
                overlap_beta / rank_query_attr, 0.0, 1.0
            )  # (B_total, B)

        w_beta = self._sharpen_weights(
            raw_incl_beta, weight_mode, threshold, tau
        )  # (B_total, B)

        # Build weighted obj target for each concept
        # X_obj_singletons: (B, n_obj, D)
        n_obj = X_obj_singletons.shape[1]
        # (B_total, B, 1, 1) * (1, B, n_obj, D) → (B_total, B, n_obj, D)
        X_obj_weighted = (
            w_beta.unsqueeze(-1).unsqueeze(-1)
            * X_obj_singletons.unsqueeze(0)
        )
        # Reshape to (B_total, B*n_obj, D) — concatenate weighted basis vectors
        X_obj_target_basis = X_obj_weighted.reshape(B_total, B * n_obj, D)
        P_obj_target = ridge_projector(
            X_obj_target_basis, lbd=self.lbd
        )  # (B_total, D, D)

        loss_beta = F.mse_loss(P_obj_all, P_obj_target)

        # ──────────────────────────────────────────────────────────
        # α direction: obj → attr
        # "Given this concept's obj subspace, which singletons are in it?"
        # v_k = Inclusion(P_obj_singleton_k ⊆ P_obj_concept)
        #     = Tr(P_obj_concept @ P_obj_singleton_k) / Tr(P_obj_singleton_k)
        # Then intersect the attr subspaces of matching singletons via
        # weighted average + power iteration.
        # ──────────────────────────────────────────────────────────

        # Tr(P_c[i] @ P_s[k]) for all (i, k) pairs
        overlap_alpha = torch.einsum(
            "aij,bji->ab", P_obj_all, P_obj_singletons
        )  # (B_total, B)
        if self.inclusion_type == "residual":
            trace_singleton_obj = torch.einsum("bii->b", P_obj_singletons).unsqueeze(0)  # (1, B)
            residual_alpha = F.relu(trace_singleton_obj - overlap_alpha)
            raw_incl_alpha = torch.clamp(
                torch.exp(-self.residual_inclusion_gamma * residual_alpha), 0.0, 1.0
            )  # (B_total, B)
        else:
            rank_singleton_obj = (
                torch.einsum("bii->b", P_obj_singletons).unsqueeze(0) + 1e-6
            )  # (1, B)
            raw_incl_alpha = torch.clamp(
                overlap_alpha / rank_singleton_obj, 0.0, 1.0
            )  # (B_total, B)

        v_alpha = self._sharpen_weights(
            raw_incl_alpha, weight_mode, threshold, tau
        )  # (B_total, B)

        # Build weighted attr intersection target
        # P_avg[i] = Σ_k v[i,k] * P_attr_singletons[k] / Σ_k v[i,k]
        # (B_total, B, 1, 1) * (1, B, D, D) → sum over B → (B_total, D, D)
        P_weighted = (
            v_alpha.unsqueeze(-1).unsqueeze(-1)
            * P_attr_singletons.unsqueeze(0)
        ).sum(dim=1)  # (B_total, D, D)
        v_sum = (
            v_alpha.sum(dim=1).unsqueeze(-1).unsqueeze(-1) + 1e-6
        )  # (B_total, 1, 1)
        P_avg = P_weighted / v_sum  # (B_total, D, D)

        # Power-iteration intersection
        P_attr_target = torch.linalg.matrix_power(
            P_avg, self.intersection_power_steps
        )  # (B_total, D, D)

        loss_alpha = F.mse_loss(P_attr_all, P_attr_target)

        return loss_beta + loss_alpha

    def compute_basis_orthogonality_loss(self, X):
        B, N, D = X.shape

        loss_rank = 0.0
        loss_ortho = 0.0
        loss_polarize = 0.0
        loss_balance = 0.0
        loss_balance_sharp = 0.0
        loss_bimodal = 0.0
        loss_bimodal_uni = 0.0
        total_loss = 0.0
        basis_orthogonality_loss_type = self.config["model"]["config"]["basis_orthogonality_loss_type"]
        default_target = float(self.config["model"]["config"].get("basis_orthogonality_rank_target", 2.0))
        balance_weight = float(self.config["model"]["config"].get("balance_weight", 1.0))

        # Hybrid anneal: interpolate between ortho and polarize over epochs
        # basis_ortho_anneal_epochs: [start, end] — ortho weight goes 1→0, polarize goes 0→1
        anneal_cfg = self.config["model"]["config"].get("basis_ortho_anneal_epochs", None)
        if anneal_cfg is not None:
            anneal_start, anneal_end = anneal_cfg
            epoch = self.current_epoch
            if epoch < anneal_start:
                ortho_alpha = 1.0
            elif epoch >= anneal_end:
                ortho_alpha = 0.0
            else:
                ortho_alpha = 1.0 - (epoch - anneal_start) / (anneal_end - anneal_start)
            polarize_alpha = 1.0 - ortho_alpha
        else:
            ortho_alpha = 1.0
            polarize_alpha = 1.0

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
            # "ortho": push all pairwise cos_sim to 0 (FLAWED — ignores shared attributes)
            loss_ortho += torch.mean(G[mask] ** 2)*10
            # "polarize": push |cos_sim| toward {0, 1} — allows aligned OR orthogonal
            # |x|*(1-|x|) = 0 at x=0 and |x|=1, max at |x|=0.5
            abs_cos = G[mask].abs()
            loss_polarize += torch.mean(abs_cos * (1 - abs_cos)) * 10
            # "balance": pull mean(|cos_sim|) toward the theoretically correct fraction
            # of aligned pairs: p_target = 1/F where F = number of fillers per slot.
            # With F fillers, each filler group has N/F singletons → C(N/F,2) same-filler
            # pairs out of C(N,2) total → p ≈ 1/F for large N.
            # Uses KL divergence from Bernoulli(p_target) for smooth penalty.
            p = abs_cos.mean().clamp(1e-6, 1 - 1e-6)
            p_target = 1.0 / max(float(rank_targets[basis_idx]), 1.0)
            p_t = max(p_target, 1e-6)
            p_t = min(p_t, 1.0 - 1e-6)
            # KL(Bernoulli(p_t) || Bernoulli(p)): zero when p = p_t
            loss_balance += (p_t * torch.log(torch.tensor(p_t, device=X.device) / p)
                             + (1 - p_t) * torch.log(torch.tensor(1 - p_t, device=X.device) / (1 - p)))
            # "balance_sharp": quadratic penalty pulling p toward p_target
            loss_balance_sharp += (p - p_target) ** 2
            # "bimodal": min(x², (1-x)²) per pair — pushes each |cos_sim| to {0, 1}
            # Unlike polarize (x*(1-x)), bimodal uses the closer target's quadratic,
            # giving steeper gradient near the targets and a sharp cusp at 0.5
            loss_bimodal += torch.mean(torch.min(abs_cos**2, (1-abs_cos)**2)) * 10
            # "bimodal_uni": x²(1-x)² on raw cos_sim (NOT abs) — roots at 0 and +1 only.
            # At cos=-1 the penalty is 4.0 (vs 0 for bimodal), preventing anti-alignment.
            # Smooth C∞ polynomial with no min() kink → cleaner gradients.
            cos_raw = G[mask]
            loss_bimodal_uni += torch.mean((cos_raw ** 2) * ((1.0 - cos_raw) ** 2)) * 10

        loss_rank /= N
        loss_ortho /= N
        loss_polarize /= N
        loss_balance /= N
        loss_balance_sharp /= N
        loss_bimodal /= N
        loss_bimodal_uni /= N

        if "rank" in basis_orthogonality_loss_type:
            total_loss += loss_rank
        if "ortho" in basis_orthogonality_loss_type:
            total_loss += loss_ortho * ortho_alpha
        if "polarize" in basis_orthogonality_loss_type:
            total_loss += loss_polarize * polarize_alpha
        if "balance" in basis_orthogonality_loss_type:
            total_loss += loss_balance * balance_weight
        if "balance_sharp" in basis_orthogonality_loss_type:
            total_loss += loss_balance_sharp * balance_weight
        if "bimodal" in basis_orthogonality_loss_type:
            total_loss += loss_bimodal
        if "bimodal_uni" in basis_orthogonality_loss_type:
            total_loss += loss_bimodal_uni

        return total_loss

    def compute_rank_complementarity_loss(self, P_attr_tensor, P_obj_tensor, singletons_mask):
        """
        Rank Complementarity: rc = Tr(P_attr)/n_attr + Tr(P_obj)/n_obj.
        In the ideal FCA lattice:
          - Singletons: attr_rank=n_attr, obj_rank=1  → rc = 1 + 1/n_obj
          - Non-singletons: by concept maximality, rc = 1.0
            (if attr shrinks, obj expands to compensate — they trade off)
        This couples attr and obj ranks per-concept, replacing batch-level
        proportionality losses with a per-concept structural invariant.
        """
        attr_ranks = torch.einsum("bii->b", P_attr_tensor)
        obj_ranks = torch.einsum("bii->b", P_obj_tensor)
        rc = attr_ranks / self.n_attr + obj_ranks / self.n_obj

        singleton_target = 1.0 + 1.0 / self.n_obj
        targets = torch.where(
            singletons_mask,
            torch.tensor(singleton_target, device=rc.device),
            torch.tensor(1.0, device=rc.device)
        )
        return F.mse_loss(rc, targets)

    def compute_margin_topological_sort_loss(self, P_attr_tensor, P_attr_singletons=None, cardinalities_tensor=None, combination_indices=None):
        """
        Margin-Based Topological Sorting: enforces rank ordering across hierarchy
        levels by comparing singletons against analytically computed intersections.

        The key insight: comparing singletons to singletons yields zero asymmetry
        (all have rank ≈ n_attr). Instead, we explicitly compute intersection
        projectors from singleton pairs and enforce that intersection rank <
        singleton rank with a margin.

        For each (singleton i, intersection j):
          - Compute inclusion asymmetry S_ij
          - Enforce: rank(singleton) - rank(intersection) >= margin * S_ij
          - Loss = mean(ReLU(margin * S - (rank_sing - rank_inter)))

        Falls back to the original all-pairs formulation if singletons/cardinalities
        are not provided.
        """
        B_total, D, _ = P_attr_tensor.shape
        if B_total < 2:
            return torch.tensor(0.0, device=P_attr_tensor.device)

        margin = float(self.config["model"]["config"].get("margin_topological_sort_margin", 0.5))

        # ── Cross-level MTS: singletons vs analytically computed intersections ──
        if P_attr_singletons is not None and cardinalities_tensor is not None and combination_indices is not None:
            B_sing = P_attr_singletons.shape[0]
            if B_sing < 2:
                return torch.tensor(0.0, device=P_attr_tensor.device)

            # Compute intersection projectors for all cardinality-2+ concepts
            inter_projectors = []
            for idx in range(len(combination_indices)):
                combo = combination_indices[idx]
                card = cardinalities_tensor[idx].item() if hasattr(cardinalities_tensor[idx], 'item') else cardinalities_tensor[idx]
                if card < 2:
                    continue
                # Get the singleton projectors for this combination
                combo_P = P_attr_singletons[combo]  # (k, D, D)
                P_avg = combo_P.mean(dim=0)  # (D, D)
                P_inter = torch.linalg.matrix_power(P_avg, self.intersection_power_steps)  # (D, D)
                inter_projectors.append(P_inter)

            if len(inter_projectors) == 0:
                return torch.tensor(0.0, device=P_attr_tensor.device)

            P_inter_tensor = torch.stack(inter_projectors)  # (N_inter, D, D)
            N_inter = P_inter_tensor.shape[0]

            sing_ranks = torch.einsum("bii->b", P_attr_singletons)  # (B_sing,)
            inter_ranks = torch.einsum("bii->b", P_inter_tensor)  # (N_inter,)

            # Cross-level inclusion: incl(inter ⊆ singleton) and incl(singleton ⊆ inter)
            # overlap[i, j] = Tr(P_inter[i] @ P_singleton[j])
            overlap_is = torch.einsum("bij,kji->bk", P_inter_tensor, P_attr_singletons)  # (N_inter, B_sing)
            incl_inter_in_sing = overlap_is / inter_ranks.clamp(min=1e-8).unsqueeze(1)  # (N_inter, B_sing)

            overlap_si = torch.einsum("bij,kji->bk", P_attr_singletons, P_inter_tensor)  # (B_sing, N_inter)
            incl_sing_in_inter = overlap_si / sing_ranks.clamp(min=1e-8).unsqueeze(1)  # (B_sing, N_inter)

            # Asymmetric inclusion: inter ⊂ singleton (positive means inter is below singleton)
            asym = torch.relu(incl_inter_in_sing - incl_sing_in_inter.t())  # (N_inter, B_sing)

            # rank_diff[i, j] = sing_rank[j] - inter_rank[i] (should be positive)
            rank_diff = sing_ranks.unsqueeze(0) - inter_ranks.unsqueeze(1)  # (N_inter, B_sing)

            # Violation: asymmetry says inter ⊂ singleton, but rank gap is insufficient
            violation = torch.relu(margin * asym - rank_diff)

            pair_mask = asym > 0.1
            if pair_mask.sum() > 0:
                return violation[pair_mask].mean()
            return torch.tensor(0.0, device=P_attr_tensor.device)

        # ── Fallback: original all-pairs formulation ──
        attr_ranks = torch.einsum("bii->b", P_attr_tensor)
        overlap = torch.einsum("bij,kji->bk", P_attr_tensor, P_attr_tensor)
        traces = attr_ranks.clamp(min=1e-8)
        inclusion = overlap / traces.unsqueeze(1)
        asym = torch.relu(inclusion - inclusion.t())
        rank_diff = attr_ranks.unsqueeze(0) - attr_ranks.unsqueeze(1)
        violation = torch.relu(margin * asym - rank_diff)
        pair_mask = (asym > 0.1) & (~torch.eye(B_total, dtype=torch.bool, device=P_attr_tensor.device))
        if pair_mask.sum() > 0:
            return violation[pair_mask].mean()
        return torch.tensor(0.0, device=P_attr_tensor.device)

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
            r = size_val // 2

            if shape == "circle":
                cv2.circle(img, center, r, color, -1)
            elif shape == "square":
                side = size_val
                cv2.rectangle(img,
                            (center[0] - side // 2, center[1] - side // 2),
                            (center[0] + side // 2, center[1] + side // 2),
                            color, -1)
            elif shape == "triangle":
                pts = np.array([
                    [center[0], center[1] - r],
                    [center[0] - int(r * np.sin(np.pi / 3)), center[1] + r // 2],
                    [center[0] + int(r * np.sin(np.pi / 3)), center[1] + r // 2],
                ], dtype=np.int32)
                cv2.fillPoly(img, [pts], color)
            elif shape == "pentagon":
                angles = [np.pi / 2 + 2 * np.pi * k / 5 for k in range(5)]
                pts = np.array([
                    [center[0] + int(r * np.cos(a)), center[1] - int(r * np.sin(a))]
                    for a in angles
                ], dtype=np.int32)
                cv2.fillPoly(img, [pts], color)
            elif shape == "star":
                r_inner = r * 2 // 5
                pts = []
                for k in range(5):
                    a_out = np.pi / 2 + 2 * np.pi * k / 5
                    pts.append([center[0] + int(r * np.cos(a_out)),
                                center[1] - int(r * np.sin(a_out))])
                    a_in = a_out + np.pi / 5
                    pts.append([center[0] + int(r_inner * np.cos(a_in)),
                                center[1] - int(r_inner * np.sin(a_in))])
                pts = np.array(pts, dtype=np.int32)
                cv2.fillPoly(img, [pts], color)

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
        annot = num_singletons * num_concepts <= 200  # disable annotations for large grids
        annot_kws = {"size": 7} if num_singletons > 10 else {}

        sns.heatmap(obj_inclusion, annot=annot, cmap="Blues", vmin=0, vmax=1,
                    xticklabels=concept_labels, yticklabels=singleton_labels, ax=axes[0],
                    annot_kws=annot_kws)
        axes[0].set_title("Object Subspace Inclusion: $P(S_{singleton}^{obj} \subseteq S_{concept}^{obj})$\\n(Expect ~1 for valid combinations)")
        axes[0].tick_params(axis='y', labelrotation=0)
        axes[0].tick_params(axis='x', labelrotation=45)
        
        sns.heatmap(attr_inclusion, annot=annot, cmap="Oranges", vmin=0, vmax=1,
                    xticklabels=concept_labels, yticklabels=singleton_labels, ax=axes[1],
                    annot_kws=annot_kws)
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

        # 6. Full Basis Vector Plot — all concepts grouped by cardinality
        ordered_names = sorted(concept_labels, key=lambda n: len(concept_defs[n]))
        slot_colors = ["#2196F3", "#FF9800"]
        dim_labels = [f"d{i}" for i in range(self.ambient_dim)]
        n_concepts_plot = len(ordered_names)

        fig_basis, axes_basis = plt.subplots(n_concepts_plot, 1,
                                              figsize=(10, 2.2 * n_concepts_plot))
        if n_concepts_plot == 1:
            axes_basis = [axes_basis]
        for i, name in enumerate(ordered_names):
            ax = axes_basis[i]
            x_attr = X_attr_concepts[name].cpu().numpy()
            if x_attr.ndim == 1:
                x_attr = x_attr[np.newaxis, :]
            card = len(concept_defs[name])
            x = np.arange(self.ambient_dim)
            bar_width = 0.35
            for basis_idx in range(x_attr.shape[0]):
                ax.bar(x + basis_idx * bar_width, x_attr[basis_idx],
                       width=bar_width, color=slot_colors[basis_idx % len(slot_colors)],
                       alpha=0.8, label=f"Slot {basis_idx}")
            if self.ambient_dim > 2:
                ax.axvline(x=self.ambient_dim / 2 - 0.5, color="gray",
                           linestyle="--", alpha=0.4, linewidth=1)
            ax.set_ylabel("Value", fontsize=9)
            ax.set_xticks(x + bar_width / 2)
            ax.set_xticklabels(dim_labels, fontsize=9)
            ax.set_title(f"{name}  [card={card}]", fontsize=11, fontweight="bold", loc="left")
            ax.axhline(y=0, color="black", linewidth=0.5)
            if i == 0:
                ax.legend(fontsize=9, loc="upper right")
        plt.suptitle(f"Attr Basis Vectors (X_attr) — All Concepts (Epoch {self.current_epoch})",
                     fontsize=14, y=1.01)
        plt.tight_layout()
        basis_tensor = self._fig_to_tensor(fig_basis)

        # 7. Rank Bar Chart
        attr_ranks_dict = {}
        obj_ranks_dict = {}
        for name in ordered_names:
            idx = concept_labels.index(name)
            attr_ranks_dict[name] = torch.trace(P_attr_concepts[idx]).item()
            obj_ranks_dict[name] = torch.trace(P_obj_concepts[idx]).item()

        card_colors = {1: "#4CAF50", 2: "#2196F3", 4: "#F44336"}
        for c in range(3, max(len(concept_defs[n]) for n in ordered_names) + 1):
            if c not in card_colors:
                card_colors[c] = "#FF9800"
        bar_colors = [card_colors.get(len(concept_defs[n]), "#999999") for n in ordered_names]
        x_pos = np.arange(len(ordered_names))

        fig_ranks, (ax_ar, ax_or) = plt.subplots(1, 2, figsize=(14, 5))

        bars1 = ax_ar.bar(x_pos, [attr_ranks_dict[n] for n in ordered_names],
                          color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax_ar.set_xticks(x_pos)
        ax_ar.set_xticklabels(ordered_names, rotation=45, ha="right", fontsize=9)
        ax_ar.set_ylabel("Rank (trace of projector)")
        ax_ar.set_title("Attribute Subspace Ranks", fontsize=13)
        ax_ar.axhline(y=self.n_attr, color="red", linestyle="--", alpha=0.5,
                      label=f"max rank = {self.n_attr}")
        ax_ar.legend(fontsize=9)
        for bar, val in zip(bars1, [attr_ranks_dict[n] for n in ordered_names]):
            ax_ar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                       f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        bars2 = ax_or.bar(x_pos, [obj_ranks_dict[n] for n in ordered_names],
                          color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax_or.set_xticks(x_pos)
        ax_or.set_xticklabels(ordered_names, rotation=45, ha="right", fontsize=9)
        ax_or.set_ylabel("Rank (trace of projector)")
        ax_or.set_title("Object Subspace Ranks", fontsize=13)
        ax_or.axhline(y=self.n_obj, color="red", linestyle="--", alpha=0.5,
                      label=f"max rank = {self.n_obj}")
        ax_or.legend(fontsize=9)
        for bar, val in zip(bars2, [obj_ranks_dict[n] for n in ordered_names]):
            ax_or.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                       f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=card_colors.get(k, "#999"), label=f"Card {k}")
                           for k in sorted(set(len(concept_defs[n]) for n in ordered_names))]
        fig_ranks.legend(handles=legend_elements, loc="upper center", ncol=len(legend_elements),
                         fontsize=10, bbox_to_anchor=(0.5, 1.02))
        plt.suptitle(f"Epoch {self.current_epoch}", fontsize=12, y=1.06)
        plt.tight_layout()
        ranks_tensor = self._fig_to_tensor(fig_ranks)

        # 8. Hasse Diagram
        hasse_tensor = self._plot_hasse_diagram(
            concept_defs, ordered_names, attr_ranks_dict, obj_ranks_dict,
            P_attr_concepts, P_obj_concepts, concept_labels
        )

        self.train()
        return heatmap_tensor, basis_tensor, ranks_tensor, hasse_tensor

    @staticmethod
    def _fig_to_tensor(fig):
        """Convert a matplotlib figure to a CHW uint8 tensor."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img_arr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_arr).permute(2, 0, 1)

    def _plot_hasse_diagram(self, concept_defs, ordered_names, attr_ranks_dict,
                            obj_ranks_dict, P_attr_concepts, P_obj_concepts,
                            concept_labels):
        """Build a Hasse diagram of the concept lattice with Galois compliance coloring."""
        # Compute NxN inclusion for attr
        N = len(ordered_names)
        attr_incl = np.zeros((N, N))
        obj_incl = np.zeros((N, N))
        for i in range(N):
            idx_i = concept_labels.index(ordered_names[i])
            for j in range(N):
                idx_j = concept_labels.index(ordered_names[j])
                attr_incl[i, j] = get_inclusion(
                    P_attr_concepts[idx_i].unsqueeze(0),
                    P_attr_concepts[idx_j].unsqueeze(0)
                ).item()
                obj_incl[i, j] = get_inclusion(
                    P_obj_concepts[idx_i].unsqueeze(0),
                    P_obj_concepts[idx_j].unsqueeze(0)
                ).item()

        # Find lattice edges (parent has strictly more objects than child)
        lattice_edges = []
        for a in ordered_names:
            for b in ordered_names:
                if a == b:
                    continue
                if set(concept_defs[b]) < set(concept_defs[a]):
                    lattice_edges.append((a, b))

        # Filter to direct (cover) edges only
        def is_direct(parent, child):
            set_p = set(concept_defs[parent])
            set_c = set(concept_defs[child])
            for mid in ordered_names:
                if mid == parent or mid == child:
                    continue
                set_m = set(concept_defs[mid])
                if set_c < set_m and set_m < set_p:
                    return False
            return True

        direct_edges = [(p, c) for p, c in lattice_edges if is_direct(p, c)]

        # Layout by cardinality
        levels = {}
        for name in ordered_names:
            card = len(concept_defs[name])
            levels.setdefault(card, []).append(name)

        sorted_cards = sorted(levels.keys(), reverse=True)
        y_map = {card: idx + 1 for idx, card in enumerate(sorted_cards)}
        max_level_width = max(len(names) for names in levels.values())
        x_spacing = max(1.2, min(2.0, 30.0 / max(max_level_width, 1)))
        positions = {}
        for card, names in levels.items():
            y = y_map[card]
            n = len(names)
            for i, name in enumerate(names):
                x = (i - (n - 1) / 2) * x_spacing
                positions[name] = (x, y)

        fig_w = max(12, max_level_width * x_spacing + 4)
        fig, ax = plt.subplots(figsize=(fig_w, 8))

        # Draw edges
        show_edge_labels = len(direct_edges) <= 30
        for parent, child in direct_edges:
            pi = ordered_names.index(parent)
            ci = ordered_names.index(child)
            attr_score = attr_incl[pi, ci]
            obj_score = obj_incl[ci, pi]
            galois_score = min(attr_score, obj_score)
            color = plt.cm.RdYlGn(galois_score)
            lw = 1.5 + 2.0 * galois_score
            x_p, y_p = positions[parent]
            x_c, y_c = positions[child]
            ax.annotate("", xy=(x_c, y_c + 0.15), xytext=(x_p, y_p - 0.15),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                        mutation_scale=15))
            if show_edge_labels:
                mid_x = (x_p + x_c) / 2
                mid_y = (y_p + y_c) / 2
                ax.text(mid_x + 0.15, mid_y, f"A:{attr_score:.2f}\nO:{obj_score:.2f}",
                        fontsize=7, ha="left", va="center", color="gray",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  alpha=0.7, edgecolor="none"))

        # Draw nodes
        card_node_colors = {1: "#81C784", 2: "#64B5F6", 4: "#EF5350"}
        for c in sorted_cards:
            if c not in card_node_colors:
                card_node_colors[c] = "#FF9800"

        label_fontsize = 7 if N > 15 else 9
        rank_fontsize = 6 if N > 15 else 7
        for name in ordered_names:
            x, y = positions[name]
            card = len(concept_defs[name])
            node_size = 300 + card * 200
            nc = card_node_colors.get(card, "#BDBDBD")
            ax.scatter(x, y, s=node_size, c=nc, edgecolors="black", linewidth=1.5, zorder=5)
            ax.text(x, y - 0.35, name, ha="center", va="top", fontsize=label_fontsize, fontweight="bold",
                    rotation=45 if N > 15 else 0)
            ar = attr_ranks_dict[name]
            orr = obj_ranks_dict[name]
            ax.text(x, y + 0.02, f"A:{ar:.1f} O:{orr:.1f}",
                    ha="center", va="center", fontsize=rank_fontsize)

        all_x = [p[0] for p in positions.values()]
        all_y = [p[1] for p in positions.values()]
        ax.set_xlim(min(all_x) - 2, max(all_x) + 2)
        ax.set_ylim(min(all_y) - 0.7, max(all_y) + 0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"Concept Lattice — Hasse Diagram (Epoch {self.current_epoch})\n"
            "Edge labels: A=attr incl(parent⊆child), O=obj incl(child⊆parent)",
            fontsize=13, pad=20)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02,
                     label="Galois compliance (min of both inclusions)")
        plt.tight_layout()
        return self._fig_to_tensor(fig)

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

        inclusion_fig, basis_fig, ranks_fig, hasse_fig = self.log_concept_lattice_inclusion()
        log_dict = {
            "lattice_inclusion_heatmap": wandb.Image(inclusion_fig),
            "concept_ranks": wandb.Image(ranks_fig),
            "concept_lattice_hasse": wandb.Image(hasse_fig),
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