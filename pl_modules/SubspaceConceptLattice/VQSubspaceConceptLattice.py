"""
MemorySubspaceConceptLattice — Memory-based compositional concept learning.

Images are encoded to per-slot attribute proposals AND object proposals in
the ambient space.  Both are resolved against persistent learnable memory
codebooks using distance-based scoring (cosine or inclusion).

Cardinality-wise combinations are formed from the batch (like the original
SubspaceConceptLattice), so we get concepts at every cardinality level.
The memory resolution ensures representations snap to discrete entries,
enabling clean compositional structure.

Losses operate on both the resolved representations (batch-level) and
the memory itself (structural).
"""
import itertools
import random
import io

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import lightning as L
import wandb

from .PerceptualEncoder import ViTEncoder
from .ConceptEncoder import MemoryConceptEncoder
from .Decoder import ViTDecoder
from .concept_utils import ridge_projector, get_inclusion


class MemorySubspaceConceptLattice(L.LightningModule):
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
        self.fillers_per_slot = self.model_config["fillers_per_slot"]

        self.max_combinations_per_cardinality = self.model_config["max_combinations_per_cardinality"]
        self.max_cardinality = self.model_config.get("max_cardinality", None)
        self.loss_weights = self.model_config["loss_weights"]

        # Loss targets: which representation each loss operates on.
        # "resolved" = post-memory (default), "proposal" = pre-memory,
        # "both" = average of both.
        default_targets = {
            "max_singleton_attr_rank_loss": "resolved",
            "repulsion_loss_attr": "resolved",
            "repulsion_loss_obj": "resolved",
            "galois_attr_loss": "resolved",
            "galois_obj_loss": "resolved",
            "intersection_consistency_loss": "resolved",
            "union_consistency_loss": "resolved",
            "attr_sink_loss": "resolved",
            "loss_attr_obj_inv_prop": "resolved",
            "cosine_repulsion_loss_attr": "resolved",
            "cosine_repulsion_loss_obj": "resolved",
            "loss_obj_rank_card_prop": "resolved",
            "loss_attr_rank_card_inv_prop": "resolved",
            "attr_obj_orthogonality_loss": "resolved",
        }
        user_targets = self.model_config.get("loss_targets", {})
        self.loss_targets = {**default_targets, **user_targets}

        # Utilization loss mode: "hard" (bincount KL) or "soft" (sharpened softmax KL)
        self.utilization_mode = self.model_config.get("utilization_mode", "hard")
        self.utilization_tau = self.model_config.get("utilization_tau", 0.1)
        self.obj_rank_loss_mode = self.model_config.get("obj_rank_loss_mode", "svd")

        # Gumbel-Softmax temperature schedule
        self.tau_start = self.model_config.get("tau_start", 1.0)
        self.tau_end = self.model_config.get("tau_end", 0.1)
        self.tau_anneal_epochs = self.model_config.get("tau_anneal_epochs", 50)

        # ── Perceptual Encoder ──────────────────────────────────────────
        perceptual_encoder_config = self.model_config["perceptual_encoder"]["config"]
        perceptual_encoder_config.update({
            "embed_dim": self.embed_dim,
            "image_size": self.image_size,
            "image_channels": self.image_channels,
        })
        self.perceptual_encoder = ViTEncoder(perceptual_encoder_config)

        # ── Memory Concept Encoder ──────────────────────────────────────
        concept_encoder_config = self.model_config["concept_encoder"]["config"]
        concept_encoder_config.update({
            "embed_dim": self.embed_dim,
            "ambient_dim": self.ambient_dim,
            "n_attr": self.n_attr,
            "n_obj": self.n_obj,
            "lbd": self.lbd,
            "fillers_per_slot": self.fillers_per_slot,
            "dims_per_slot": self.model_config.get("dims_per_slot", None),
            "resolution_method": self.model_config.get("resolution_method", "argmax"),
            "resolution_tau": self.model_config.get("resolution_tau", 1.0),
            "commitment_codebook_weight": self.model_config.get("commitment_codebook_weight", 1.0),
            "commitment_encoder_weight": self.model_config.get("commitment_encoder_weight", 1.0),
            "separate_obj_attr_attn": self.model_config.get("separate_obj_attr_attn", False),
            "separate_slot_attn": self.model_config.get("separate_slot_attn", False),
            "attn_depth": self.model_config.get("attn_depth", 1),
            "attn_heads": self.model_config.get("attn_heads", None),
            "attn_mlp_ratio": self.model_config.get("attn_mlp_ratio", 4.0),
            # Legacy compat: pass through old keys too
            "per_slot_attention": self.model_config.get("per_slot_attention", False),
            "slot_attention_depth": self.model_config.get("slot_attention_depth", 1),
            "obj_attn_depth": self.model_config.get("obj_attn_depth", 1),
            "obj_attn_heads": self.model_config.get("obj_attn_heads", None),
            "obj_attn_mlp_ratio": self.model_config.get("obj_attn_mlp_ratio", 4.0),
        })
        self.concept_encoder = MemoryConceptEncoder(concept_encoder_config)

        # ── Decoder ─────────────────────────────────────────────────────
        decoder_config = self.model_config["decoder"]["config"]
        decoder_config.update({
            "embed_dim": self.embed_dim,
            "image_size": self.image_size,
            "image_channels": self.image_channels,
        })
        self.decoder = ViTDecoder(decoder_config)

        # ── Attr-Obj Orthogonality Bridge ────────────────────────────────
        # Learnable orthogonal matrix W such that P_obj @ (W P_attr W^T) ≈ 0.
        # This forces attr and obj subspaces to capture complementary info.
        if self.loss_weights.get("attr_obj_orthogonality_loss", 0) > 0:
            W_linear = nn.Linear(self.ambient_dim, self.ambient_dim, bias=False)
            self.W = torch.nn.utils.parametrizations.orthogonal(W_linear)
        else:
            self.W = None

        self.viz_datapoint = None

    # ================================================================== #
    #  Temperature schedule
    # ================================================================== #
    def get_tau(self):
        if self.tau_anneal_epochs <= 0:
            return self.tau_end
        progress = min(self.current_epoch / self.tau_anneal_epochs, 1.0)
        return self.tau_start + progress * (self.tau_end - self.tau_start)

    # ================================================================== #
    #  Forward — cardinality-wise combinations
    # ================================================================== #
    def forward(self, x):
        images = x["images"]  # (B, 3, H, W)
        B = images.shape[0]
        representations = self.perceptual_encoder(images)  # (B, P, E)

        tau = self.get_tau()
        hard = not self.training

        # Accumulators for resolved and proposal tensors
        all_resolved_X_attr, all_resolved_X_obj = [], []
        all_resolved_P_attr, all_resolved_P_obj = [], []
        all_proposal_X_attr, all_proposal_X_obj = [], []
        all_proposal_P_attr, all_proposal_P_obj = [], []
        all_cardinalities = []
        all_combination_indices = []
        all_attr_assign_idx = []
        all_obj_assign_idx = []
        singleton_attr_assign_idx = []
        singleton_obj_assign_idx = []
        singleton_attr_scores = []
        singleton_obj_scores = []
        nonsingleton_attr_scores = []
        nonsingleton_obj_scores = []
        total_commitment_loss = 0.0
        total_proposal_norm_loss = 0.0
        n_groups = 0

        max_k = min(B, self.max_cardinality) if self.max_cardinality else B
        for k in range(1, max_k + 1):
            combination_indices = list(itertools.combinations(range(B), k))
            if len(combination_indices) > self.max_combinations_per_cardinality and k > 1:
                combination_indices = random.sample(
                    combination_indices, self.max_combinations_per_cardinality
                )

            combo_tensor = torch.tensor(combination_indices, device=self.device)
            R_subset = representations[combo_tensor]  # (num_combos, k, P, E)

            is_singleton = (k == 1)
            enc_out = self.concept_encoder(
                R_subset, tau=tau, hard=hard, singleton=is_singleton
            )

            all_resolved_X_attr.append(enc_out["resolved_X_attr"])
            all_resolved_X_obj.append(enc_out["resolved_X_obj"])
            all_resolved_P_attr.append(enc_out["resolved_P_attr"])
            all_resolved_P_obj.append(enc_out["resolved_P_obj"])
            all_proposal_X_attr.append(enc_out["proposal_X_attr"])
            all_proposal_X_obj.append(enc_out["proposal_X_obj"])
            all_proposal_P_attr.append(enc_out["proposal_P_attr"])
            all_proposal_P_obj.append(enc_out["proposal_P_obj"])
            all_cardinalities.append(
                torch.full((len(combination_indices),), k, device=self.device)
            )
            all_combination_indices.extend(combo_tensor)
            all_attr_assign_idx.append(enc_out["attr_assign_idx"])
            all_obj_assign_idx.append(enc_out["obj_assign_idx"])
            if is_singleton:
                singleton_attr_assign_idx.append(enc_out["attr_assign_idx"])
                singleton_obj_assign_idx.append(enc_out["obj_assign_idx"])
                singleton_attr_scores.append(enc_out["attr_scores"])
                singleton_obj_scores.append(enc_out["obj_scores"])
            else:
                nonsingleton_attr_scores.append(enc_out["attr_scores"])
                nonsingleton_obj_scores.append(enc_out["obj_scores"])
            total_commitment_loss += enc_out["commitment_loss"]
            total_proposal_norm_loss += enc_out["proposal_norm_loss"]
            n_groups += 1

        cardinalities = torch.cat(all_cardinalities, dim=0)
        commitment_loss = total_commitment_loss / max(1, n_groups)
        proposal_norm_loss = total_proposal_norm_loss / max(1, n_groups)

        # Concatenate resolved tensors
        resolved_X_attr = torch.cat(all_resolved_X_attr, dim=0)
        resolved_X_obj = torch.cat(all_resolved_X_obj, dim=0)
        resolved_P_attr = torch.cat(all_resolved_P_attr, dim=0)
        resolved_P_obj = torch.cat(all_resolved_P_obj, dim=0)

        # Concatenate proposal tensors
        proposal_X_attr = torch.cat(all_proposal_X_attr, dim=0)
        proposal_X_obj = torch.cat(all_proposal_X_obj, dim=0)
        proposal_P_attr = torch.cat(all_proposal_P_attr, dim=0)
        proposal_P_obj = torch.cat(all_proposal_P_obj, dim=0)

        attr_assign_idx = torch.cat(all_attr_assign_idx, dim=0)
        obj_assign_idx = torch.cat(all_obj_assign_idx, dim=0)
        singleton_attr_idx = torch.cat(singleton_attr_assign_idx, dim=0)
        singleton_obj_idx = torch.cat(singleton_obj_assign_idx, dim=0)
        singleton_attr_scores_cat = torch.cat(singleton_attr_scores, dim=0)
        singleton_obj_scores_cat = torch.cat(singleton_obj_scores, dim=0)
        nonsingleton_attr_scores_cat = (
            torch.cat(nonsingleton_attr_scores, dim=0)
            if nonsingleton_attr_scores
            else torch.zeros(0, self.concept_encoder.n_attr_candidates, device=self.device)
        )
        nonsingleton_obj_scores_cat = (
            torch.cat(nonsingleton_obj_scores, dim=0)
            if nonsingleton_obj_scores
            else torch.zeros(0, self.concept_encoder.n_obj_candidates, device=self.device)
        )

        singletons_mask = (cardinalities == 1)
        X_attr_singletons = resolved_X_attr[singletons_mask]
        X_attr_dec = self.concept_encoder.attr_dec(X_attr_singletons)
        reconstructed = self.decoder(X_attr_dec)

        return {
            "images": images,
            "reconstructed_images": reconstructed,
            "representations": representations,
            # Resolved (post-memory) tensors
            "resolved_X_attr": resolved_X_attr,
            "resolved_X_obj": resolved_X_obj,
            "resolved_P_attr": resolved_P_attr,
            "resolved_P_obj": resolved_P_obj,
            # Proposal (pre-memory) tensors
            "proposal_X_attr": proposal_X_attr,
            "proposal_X_obj": proposal_X_obj,
            "proposal_P_attr": proposal_P_attr,
            "proposal_P_obj": proposal_P_obj,
            # Metadata
            "cardinalities": cardinalities,
            "combination_indices": all_combination_indices,
            "singletons_mask": singletons_mask,
            "attr_assign_idx": attr_assign_idx,
            "obj_assign_idx": obj_assign_idx,
            "singleton_attr_assign_idx": singleton_attr_idx,
            "singleton_obj_assign_idx": singleton_obj_idx,
            "singleton_attr_scores": singleton_attr_scores_cat,
            "singleton_obj_scores": singleton_obj_scores_cat,
            "nonsingleton_attr_scores": nonsingleton_attr_scores_cat,
            "nonsingleton_obj_scores": nonsingleton_obj_scores_cat,
            "commitment_loss": commitment_loss,
            "proposal_norm_loss": proposal_norm_loss,
        }

    # ================================================================== #
    #  Structural loss computation (parameterized by representation)
    # ================================================================== #
    def _compute_structural_losses(self, P_attr, P_obj, X_attr, X_obj,
                                   cardinalities, combination_indices,
                                   singletons_mask, B, B_total):
        """
        Compute all switchable structural losses on a given set of
        projectors/basis vectors. Called once per target mode (resolved
        and/or proposal).

        Returns a dict of {loss_name: scalar_tensor}.
        """
        P_attr_singletons = P_attr[singletons_mask]
        P_obj_singletons = P_obj[singletons_mask]
        X_attr_singletons = X_attr[singletons_mask]
        X_obj_singletons = X_obj[singletons_mask]

        # ── Singleton attr rank ────────────────────────────────────────
        singleton_attr_ranks = torch.einsum("bii->b", P_attr_singletons)
        max_singleton_attr_rank_loss = -torch.mean(singleton_attr_ranks)

        # ── Repulsion losses (within batch singletons) ─────────────────
        repulsion_loss_attr = self._compute_soft_repulsion(P_attr_singletons)
        repulsion_loss_obj = self._compute_soft_repulsion(P_obj_singletons)
        cosine_repulsion_loss_attr = self._compute_cosine_repulsion(X_attr_singletons)
        cosine_repulsion_loss_obj = self._compute_cosine_repulsion(X_obj_singletons)

        # ── Combination losses (Galois, IC, UC, sink) ──────────────────
        total_galois_attr_loss = 0.0
        total_galois_obj_loss = 0.0
        total_intersection_consistency_loss = 0.0
        total_union_consistency_loss = 0.0
        total_attr_sink_loss = 0.0
        total_comparisons = 0

        obj_ranks = torch.einsum("bii->b", P_obj)
        attr_ranks = torch.einsum("bii->b", P_attr)

        for cardinality in range(2, B + 1):
            card_mask = (cardinalities == cardinality)
            if not card_mask.any():
                continue

            card_combo_indices = torch.stack(
                [combination_indices[i] for i in torch.where(card_mask)[0]], dim=0
            ).to(self.device)
            num_combos = card_combo_indices.shape[0]

            card_P_attr = P_attr[card_mask]
            card_P_obj = P_obj[card_mask]

            # ── Attr sink loss (obj_rank-weighted) ─────────────────────
            card_obj_ranks = obj_ranks[card_mask]
            max_obj_rank = float(self.n_obj)
            sink_weight = (card_obj_ranks / max_obj_rank).clamp(0, 1)
            card_attr_ranks = attr_ranks[card_mask]
            total_attr_sink_loss += torch.sum((card_attr_ranks * sink_weight) ** 2)

            # ── Galois losses ──────────────────────────────────────────
            combo_P_attr_singletons = P_attr_singletons[card_combo_indices]
            combo_P_obj_singletons = P_obj_singletons[card_combo_indices]

            P_attr_comb_exp = card_P_attr.unsqueeze(1).expand(-1, cardinality, -1, -1)
            P_obj_comb_exp = card_P_obj.unsqueeze(1).expand(-1, cardinality, -1, -1)

            D = self.ambient_dim
            P_attr_c_flat = P_attr_comb_exp.reshape(-1, D, D)
            P_attr_s_flat = combo_P_attr_singletons.reshape(-1, D, D)
            P_obj_c_flat = P_obj_comb_exp.reshape(-1, D, D)
            P_obj_s_flat = combo_P_obj_singletons.reshape(-1, D, D)

            galois_attr_inc = get_inclusion(P_sub=P_attr_c_flat, P_super=P_attr_s_flat)
            galois_obj_inc = get_inclusion(P_sub=P_obj_s_flat, P_super=P_obj_c_flat)

            galois_attr_inc = galois_attr_inc.clamp(1e-6, 1.0 - 1e-6)
            galois_obj_inc = galois_obj_inc.clamp(1e-6, 1.0 - 1e-6)

            total_galois_attr_loss += F.binary_cross_entropy(
                galois_attr_inc, torch.ones_like(galois_attr_inc), reduction="sum"
            )
            total_galois_obj_loss += F.binary_cross_entropy(
                galois_obj_inc, torch.ones_like(galois_obj_inc), reduction="sum"
            )
            total_comparisons += num_combos * cardinality

            # ── Intersection consistency (attr) — power iteration ──────
            combo_P_attr_singletons_avg = combo_P_attr_singletons.mean(dim=1)
            P_attr_inter = torch.linalg.matrix_power(combo_P_attr_singletons_avg, 5)
            total_intersection_consistency_loss += F.mse_loss(
                card_P_attr, P_attr_inter, reduction="sum"
            )

            # ── Union consistency (obj) ────────────────────────────────
            combo_X_obj_singletons = X_obj_singletons[card_combo_indices].detach()
            X_obj_union_basis = combo_X_obj_singletons.view(
                num_combos, cardinality * self.n_obj, self.ambient_dim
            )
            P_obj_union = ridge_projector(X_obj_union_basis, lbd=self.lbd)
            total_union_consistency_loss += F.mse_loss(
                card_P_obj, P_obj_union, reduction="sum"
            )

        # Normalize combination losses
        n_non_singletons = max(1, B_total - B)
        attr_sink_loss = total_attr_sink_loss / n_non_singletons
        intersection_consistency_loss = total_intersection_consistency_loss / n_non_singletons
        union_consistency_loss = total_union_consistency_loss / n_non_singletons
        galois_attr_loss = total_galois_attr_loss / max(1, total_comparisons)
        galois_obj_loss = total_galois_obj_loss / max(1, total_comparisons)

        # ── Rank proportionality losses ────────────────────────────────
        loss_attr_obj_inv_prop = self._proportionality_loss(
            attr_ranks, obj_ranks, inverse=True
        )
        # Rank-cardinality proportionality:
        #   obj rank should increase with cardinality (proportional)
        #   attr rank should decrease with cardinality (inverse proportional)
        card_float = cardinalities.float()
        loss_obj_rank_card_prop = self._proportionality_loss(
            obj_ranks, card_float, inverse=False
        )
        loss_attr_rank_card_inv_prop = self._proportionality_loss(
            attr_ranks, card_float, inverse=True
        )

        # ── Attr-Obj Orthogonality Loss ───────────────────────────────────
        # Forces P_obj and W P_attr W^T to be orthogonal, pushing attr and
        # obj subspaces to encode complementary information.
        # Currently hardcoded to singletons only.
        if self.W is not None and singletons_mask.any():
            W_mat = self.W.weight  # (D, D), orthogonal
            P_attr_s = P_attr[singletons_mask]
            P_obj_s = P_obj[singletons_mask]
            P_attr_trans = torch.einsum("ij,bjk,lk->bil", W_mat, P_attr_s, W_mat)
            overlap = torch.einsum("bij,bji->b", P_obj_s, P_attr_trans)
            attr_obj_orthogonality_loss = overlap.mean()
        else:
            attr_obj_orthogonality_loss = torch.tensor(0.0, device=P_attr.device)

        return {
            "max_singleton_attr_rank_loss": max_singleton_attr_rank_loss,
            "repulsion_loss_attr": repulsion_loss_attr,
            "repulsion_loss_obj": repulsion_loss_obj,
            "cosine_repulsion_loss_attr": cosine_repulsion_loss_attr,
            "cosine_repulsion_loss_obj": cosine_repulsion_loss_obj,
            "galois_attr_loss": galois_attr_loss,
            "galois_obj_loss": galois_obj_loss,
            "intersection_consistency_loss": intersection_consistency_loss,
            "union_consistency_loss": union_consistency_loss,
            "attr_sink_loss": attr_sink_loss,
            "loss_attr_obj_inv_prop": loss_attr_obj_inv_prop,
            "loss_obj_rank_card_prop": loss_obj_rank_card_prop,
            "loss_attr_rank_card_inv_prop": loss_attr_rank_card_inv_prop,
            "attr_obj_orthogonality_loss": attr_obj_orthogonality_loss,
            # Also return ranks for logging (not used in loss, just for metrics)
            "_attr_ranks": attr_ranks,
            "_obj_ranks": obj_ranks,
        }

    # ================================================================== #
    #  Training step
    # ================================================================== #
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        images = outputs["images"]
        reconstructed = outputs["reconstructed_images"]
        B = images.shape[0]

        cardinalities = outputs["cardinalities"]
        combination_indices = outputs["combination_indices"]
        singletons_mask = outputs["singletons_mask"]
        commitment_loss = outputs["commitment_loss"]
        proposal_norm_loss = outputs["proposal_norm_loss"]
        B_total = outputs["resolved_P_attr"].shape[0]

        if self.viz_datapoint is None:
            n = min(4, B)
            self.viz_datapoint = {
                "original_images": images[:n].detach().cpu(),
                "reconstructed_images": reconstructed[:n].detach().cpu(),
            }

        # ── Reconstruction loss ─────────────────────────────────────────
        reconstruction_loss = F.mse_loss(reconstructed, images, reduction="sum") / B

        # ── Memory structural losses ────────────────────────────────────
        memory_slot_orthogonality_loss = self._memory_slot_orthogonality_loss()
        memory_obj_rank_loss = self._memory_obj_rank_loss()
        memory_obj_orthogonality_loss = self._memory_obj_orthogonality_loss()
        memory_linkage_loss = self._memory_linkage_loss()

        # ── Compute structural losses on resolved and/or proposal ───────
        shared_args = (cardinalities, combination_indices, singletons_mask, B, B_total)

        resolved_losses = self._compute_structural_losses(
            outputs["resolved_P_attr"], outputs["resolved_P_obj"],
            outputs["resolved_X_attr"], outputs["resolved_X_obj"],
            *shared_args,
        )
        proposal_losses = self._compute_structural_losses(
            outputs["proposal_P_attr"], outputs["proposal_P_obj"],
            outputs["proposal_X_attr"], outputs["proposal_X_obj"],
            *shared_args,
        )

        # Select each loss based on its target mode
        switchable_names = [
            "max_singleton_attr_rank_loss", "repulsion_loss_attr",
            "repulsion_loss_obj", "cosine_repulsion_loss_attr",
            "cosine_repulsion_loss_obj", "galois_attr_loss", "galois_obj_loss",
            "intersection_consistency_loss", "union_consistency_loss",
            "attr_sink_loss", "loss_attr_obj_inv_prop",
            "loss_obj_rank_card_prop", "loss_attr_rank_card_inv_prop",
            "attr_obj_orthogonality_loss",
        ]
        structural_losses = {}
        for name in switchable_names:
            target = self.loss_targets[name]
            if target == "resolved":
                structural_losses[name] = resolved_losses[name]
            elif target == "proposal":
                structural_losses[name] = proposal_losses[name]
            elif target == "both":
                structural_losses[name] = (
                    resolved_losses[name] + proposal_losses[name]
                ) / 2.0

        # ── VQ losses ───────────────────────────────────────────────────
        if self.utilization_mode == "soft":
            utilization_loss_attr, utilization_loss_obj = self._utilization_loss_soft(
                outputs["singleton_attr_scores"], outputs["singleton_obj_scores"]
            )
        else:
            utilization_loss_attr, utilization_loss_obj = self._utilization_loss_hard(
                outputs["singleton_attr_assign_idx"], outputs["singleton_obj_assign_idx"]
            )

        # All-cardinality utilization losses (soft, differentiable)
        util_anti_collapse_attr, util_anti_collapse_obj = \
            self._utilization_loss_anti_collapse(
                outputs["singleton_attr_scores"], outputs["singleton_obj_scores"],
                outputs["nonsingleton_attr_scores"], outputs["nonsingleton_obj_scores"],
            )
        util_rank_cond_attr, util_rank_cond_obj = \
            self._utilization_loss_rank_conditioned(
                outputs["singleton_attr_scores"], outputs["singleton_obj_scores"],
                outputs["nonsingleton_attr_scores"], outputs["nonsingleton_obj_scores"],
            )
        assignment_sharpness_attr, assignment_sharpness_obj = \
            self._assignment_sharpness_loss(
                outputs["singleton_attr_scores"], outputs["singleton_obj_scores"],
            )

        # ── Total loss ──────────────────────────────────────────────────
        w = self.loss_weights
        total_loss = (
            w["reconstruction_loss"] * reconstruction_loss
            + w["commitment_loss"] * commitment_loss
            + w["memory_slot_orthogonality_loss"] * memory_slot_orthogonality_loss
            + w["memory_obj_rank_loss"] * memory_obj_rank_loss
            + w["memory_obj_orthogonality_loss"] * memory_obj_orthogonality_loss
            + w.get("utilization_loss_attr", w.get("utilization_loss", 0)) * utilization_loss_attr
            + w.get("utilization_loss_obj", w.get("utilization_loss", 0)) * utilization_loss_obj
            + w.get("proposal_norm_loss", 0) * proposal_norm_loss
            + w.get("utilization_anti_collapse_attr", 0) * util_anti_collapse_attr
            + w.get("utilization_anti_collapse_obj", 0) * util_anti_collapse_obj
            + w.get("utilization_rank_conditioned_attr", 0) * util_rank_cond_attr
            + w.get("utilization_rank_conditioned_obj", 0) * util_rank_cond_obj
            + w.get("assignment_sharpness_attr", 0) * assignment_sharpness_attr
            + w.get("assignment_sharpness_obj", 0) * assignment_sharpness_obj
            + w.get("memory_linkage_loss", 0) * memory_linkage_loss
        )
        for name in switchable_names:
            total_loss = total_loss + w.get(name, 0) * structural_losses[name]

        # Use resolved ranks for logging (always available)
        attr_ranks = resolved_losses["_attr_ranks"]
        obj_ranks = resolved_losses["_obj_ranks"]

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": commitment_loss,
            "memory_slot_orthogonality_loss": memory_slot_orthogonality_loss,
            "memory_obj_rank_loss": memory_obj_rank_loss,
            "memory_obj_orthogonality_loss": memory_obj_orthogonality_loss,
            "utilization_loss_attr": utilization_loss_attr,
            "utilization_loss_obj": utilization_loss_obj,
            "utilization_anti_collapse_attr": util_anti_collapse_attr,
            "utilization_anti_collapse_obj": util_anti_collapse_obj,
            "utilization_rank_conditioned_attr": util_rank_cond_attr,
            "utilization_rank_conditioned_obj": util_rank_cond_obj,
            "assignment_sharpness_attr": assignment_sharpness_attr,
            "assignment_sharpness_obj": assignment_sharpness_obj,
            "memory_linkage_loss": memory_linkage_loss,
            "proposal_norm_loss": proposal_norm_loss,
            "total_loss": total_loss,
            "tau": self.get_tau(),
        }
        loss_dict.update(structural_losses)

        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)

        # Log per-cardinality ranks (resolved for consistency)
        for card in range(1, B + 1):
            mask = (cardinalities == card)
            if mask.any():
                self.log(f"rank/attr_cardinality_{card}",
                         attr_ranks[mask].mean(), on_epoch=True, prog_bar=False)
                self.log(f"rank/obj_cardinality_{card}",
                         obj_ranks[mask].mean(), on_epoch=True, prog_bar=False)

        return total_loss

    # ================================================================== #
    #  Memory structural losses
    # ================================================================== #
    def _memory_slot_orthogonality_loss(self):
        """Fillers within each slot should be orthogonal to each other."""
        loss = 0.0
        for slot_idx in range(self.n_attr):
            M = self.concept_encoder.attr_memory[slot_idx]  # (K, d)
            if M.shape[0] < 2:
                continue
            M_norm = F.normalize(M, dim=-1)
            sim = M_norm @ M_norm.t()  # (K, K)
            mask = ~torch.eye(M.shape[0], dtype=torch.bool, device=M.device)
            loss = loss + (sim[mask] ** 2).mean()
        return loss / max(1, self.n_attr)

    def _memory_obj_rank_loss(self):
        """Each object memory entry should be rank-1 (a single direction).
        Two modes controlled by self.obj_rank_loss_mode:
          "svd"   — rank = sum(s^2 / (s^2 + lbd)), avoids ridge_projector
          "trace" — rank = Tr(ridge_projector(X)), same metric as downstream logging
        """
        M = self.concept_encoder.obj_memory  # (n_obj, n_obj, ambient_dim)
        target_rank = float(1)
        loss = 0.0
        for i in range(self.n_obj):
            X = M[i]  # (n_obj, ambient_dim)
            if self.obj_rank_loss_mode == "trace":
                P = ridge_projector(X.unsqueeze(0), lbd=self.lbd).squeeze(0)
                rank = torch.trace(P)
            else:  # svd
                s = torch.linalg.svdvals(X)  # (min(n_obj, D),)
                rank = (s * s / (s * s + self.lbd)).sum()
            loss += F.mse_loss(rank, torch.tensor(target_rank, device=M.device))
        return loss / self.n_obj

    def _memory_obj_orthogonality_loss(self):
        """Object memory subspaces should be orthogonal to each other.
        Uses cross-Gram matrix: ||X_i @ X_j^T||_F^2 / (||X_i||_F * ||X_j||_F).
        Avoids ridge_projector for gradient stability."""
        M = self.concept_encoder.obj_memory  # (n_obj, n_obj, ambient_dim)
        if self.n_obj < 2:
            return torch.tensor(0.0, device=M.device)
        # Compute pairwise cross-Gram overlap
        loss = 0.0
        count = 0
        for i in range(self.n_obj):
            for j in range(i + 1, self.n_obj):
                cross = M[i] @ M[j].t()  # (n_obj, n_obj)
                norm_i = M[i].norm() + 1e-6
                norm_j = M[j].norm() + 1e-6
                loss += (cross / (norm_i * norm_j)).pow(2).sum()
                count += 1
        return loss / max(1, count)

    def _memory_linkage_loss(self):
        """Contrastive loss anchoring obj and attr memory entries via W.

        Uses a hardcoded incidence mask for the v0 dataset (4 objects, 4 fillers).
        Objects: [RedCircle, RedSquare, BlueCircle, BlueSquare]
        Fillers: [Slot0_F0=Red, Slot0_F1=Blue, Slot1_F0=Circle, Slot1_F1=Square]
        mask[i,j]=0 means obj i HAS attr j (push overlap to 0).
        mask[i,j]=1 means obj i DOES NOT HAVE attr j (push overlap above margin).
        """
        if self.W is None:
            return torch.tensor(0.0, device=self.device)

        W_mat = self.W.weight  # (D, D)

        # Object projectors from base memory entries (rank-1, first basis vec)
        base_objs_X = self.concept_encoder.obj_memory[:, 0:1, :]  # (4, 1, D)
        P_obj = ridge_projector(base_objs_X, lbd=self.lbd)  # (4, D, D)

        # Attr fillers live in per-slot dims; embed into full ambient_dim
        # by zero-padding into their coordinate mask positions.
        n_attr = self.concept_encoder.n_attr
        ambient_dim = self.concept_encoder.ambient_dim
        all_filler_vecs = []
        offset = 0
        for slot_idx in range(n_attr):
            mem = self.concept_encoder.attr_memory[slot_idx]  # (K, dims_per_slot)
            d = mem.shape[1]
            for k in range(mem.shape[0]):
                full_vec = torch.zeros(ambient_dim, device=mem.device)
                full_vec[offset:offset + d] = mem[k]
                all_filler_vecs.append(full_vec)
            offset += d
        # (n_total_fillers, ambient_dim) — each as a rank-1 basis
        filler_basis = torch.stack(all_filler_vecs).unsqueeze(1)  # (4, 1, D)
        P_attr = ridge_projector(filler_basis, lbd=self.lbd)  # (4, D, D)

        # Translate attr projectors: P_trans = W @ P_attr @ W^T
        P_attr_trans = torch.einsum("ij,ajk,lk->ail", W_mat, P_attr, W_mat)

        # Overlap matrix: (n_obj, n_fillers)
        overlap = torch.einsum("oij,aji->oa", P_obj, P_attr_trans)

        # Hardcoded incidence mask for v0 dataset
        # 1 = obj has attribute (push overlap high), 0 = does not have (push to 0)
        incidence = torch.tensor([
            [1, 0, 1, 0],  # Red Circle:  has Red, has Circle
            [1, 0, 0, 1],  # Red Square:  has Red, has Square
            [0, 1, 1, 0],  # Blue Circle: has Blue, has Circle
            [0, 1, 0, 1],  # Blue Square: has Blue, has Square
        ], device=self.device, dtype=torch.float32)

        margin = 0.5
        # Where incidence=1 (HAS): push overlap above margin
        loss_has = incidence * F.relu(margin - overlap)
        # Where incidence=0 (NOT HAS): push overlap to 0
        loss_has_not = (1 - incidence) * (overlap ** 2)

        return (loss_has + loss_has_not).mean()

    # ================================================================== #
    #  Batch-level losses
    # ================================================================== #
    def _compute_soft_repulsion(self, P_singletons):
        """Minimize pairwise overlap of singleton projectors (inclusion-based)."""
        B = P_singletons.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=P_singletons.device)
        P_vec = P_singletons.reshape(B, -1)
        overlap = P_vec @ P_vec.t()
        mask = ~torch.eye(B, dtype=torch.bool, device=P_singletons.device)
        return overlap[mask].mean()

    def _compute_cosine_repulsion(self, X_singletons):
        """
        Minimize pairwise cosine similarity of singleton basis vectors.
        Operates on X (B, n_basis, ambient_dim) rather than projectors.

        Computes cosine similarity per basis vector separately, then averages.
        This preserves the structural meaning of each basis (e.g., color slot
        vs shape slot) instead of mixing them via flattening.
        """
        B, n_basis, D = X_singletons.shape
        if B < 2:
            return torch.tensor(0.0, device=X_singletons.device)
        mask = ~torch.eye(B, dtype=torch.bool, device=X_singletons.device)
        total = 0.0
        for basis_idx in range(n_basis):
            vecs = X_singletons[:, basis_idx, :]  # (B, D)
            normed = F.normalize(vecs, p=2, dim=-1)
            cos_sim = normed @ normed.t()  # (B, B)
            total = total + cos_sim[mask].pow(2).mean()
        return total / n_basis

    def _rank_conservation_loss(self, P_attr_singletons):
        """
        For each pair of singleton attr subspaces, enforce:
          rank(intersection) + rank(residual_i) = rank(original_i)

        Uses projector-based intersection: P_inter ≈ 0.5*(P_i@P_j + P_j@P_i).
        Residual: P_res_i = P_i - P_inter.
        """
        B = P_attr_singletons.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=P_attr_singletons.device)

        total_loss = 0.0
        num_pairs = 0

        for i in range(B):
            for j in range(i + 1, B):
                P_i = P_attr_singletons[i]
                P_j = P_attr_singletons[j]

                P_inter = 0.5 * (P_i @ P_j + P_j @ P_i)
                P_res_i = P_i - P_inter
                P_res_j = P_j - P_inter

                rank_i = torch.trace(P_i)
                rank_j = torch.trace(P_j)
                rank_inter = torch.trace(P_inter)
                rank_res_i = torch.trace(P_res_i)
                rank_res_j = torch.trace(P_res_j)

                total_loss += (rank_inter + rank_res_i - rank_i) ** 2
                total_loss += (rank_inter + rank_res_j - rank_j) ** 2
                num_pairs += 1

        return total_loss / max(1, num_pairs)

    def _proportionality_loss(self, x, y, inverse=False, eps=1e-8):
        """Pearson correlation-based proportionality loss."""
        x_c = x - x.mean()
        y_c = y - y.mean()
        covar = (x_c * y_c).sum()
        var_x = (x_c ** 2).sum()
        var_y = (y_c ** 2).sum()
        # Add eps inside sqrt to prevent grad(sqrt(0)) = inf
        corr = covar / (torch.sqrt(var_x * var_y + eps) + eps)
        if inverse:
            return 1.0 + corr
        return 1.0 - corr

    # ================================================================== #
    #  VQ losses
    # ================================================================== #
    def _utilization_loss_hard(self, attr_assign_idx, obj_assign_idx):
        """
        Utilization loss on hard assignments via KL divergence from uniform.
        No gradient through encoder — provides honest penalty signal.
        """
        C_attr = self.concept_encoder.n_singleton_attr_candidates
        counts = torch.bincount(attr_assign_idx, minlength=C_attr).float() + 1e-6
        emp = counts / counts.sum()
        uni = torch.ones(C_attr, device=counts.device) / C_attr
        loss_attr = (emp * (emp.log() - uni.log())).sum()

        C_obj = self.concept_encoder.n_singleton_obj_candidates
        counts = torch.bincount(obj_assign_idx, minlength=C_obj).float() + 1e-6
        emp = counts / counts.sum()
        uni = torch.ones(C_obj, device=counts.device) / C_obj
        loss_obj = (emp * (emp.log() - uni.log())).sum()

        return loss_attr, loss_obj

    def _utilization_loss_soft(self, attr_scores, obj_scores):
        """
        Utilization loss using sharpened softmax on inclusion scores.
        Temperature self.utilization_tau controls sharpness (lower = sharper).
        Differentiable — gradients flow through scores to encoder and memory.
        """
        losses = []
        for scores in [attr_scores, obj_scores]:
            soft_probs = F.softmax(scores / self.utilization_tau, dim=-1)  # (B, C)
            avg_usage = soft_probs.mean(dim=0)  # (C,)
            C = scores.shape[1]
            losses.append((avg_usage * (avg_usage * C + 1e-8).log()).sum())
        return losses[0], losses[1]

    def _utilization_loss_anti_collapse(self, singleton_attr_scores,
                                       singleton_obj_scores,
                                       nonsingleton_attr_scores,
                                       nonsingleton_obj_scores):
        """
        Anti-collapse utilization: encourage all candidates to receive some
        soft probability mass across ALL cardinalities.

        Uses sharpened softmax on inclusion scores (differentiable).
        Singletons and non-singletons use different candidate sets, so we
        compute soft usage independently and combine.

        Loss = mean(-log(avg_usage_i)) over all candidates.
        """
        tau = self.utilization_tau

        def _soft_anti_collapse(scores, label=""):
            """Compute anti-collapse for one set of scores (B, C)."""
            if scores.shape[0] == 0:
                return torch.tensor(0.0, device=scores.device)
            soft_probs = F.softmax(scores / tau, dim=-1)  # (B, C)
            avg_usage = soft_probs.mean(dim=0)  # (C,)
            return -(avg_usage + 1e-8).log().mean()

        # Singletons use reduced candidate set; non-singletons use full set
        loss_attr_s = _soft_anti_collapse(singleton_attr_scores)
        loss_attr_ns = _soft_anti_collapse(nonsingleton_attr_scores)
        n_parts = (singleton_attr_scores.shape[0] > 0) + (nonsingleton_attr_scores.shape[0] > 0)
        loss_attr = (loss_attr_s + loss_attr_ns) / max(1, n_parts)

        loss_obj_s = _soft_anti_collapse(singleton_obj_scores)
        loss_obj_ns = _soft_anti_collapse(nonsingleton_obj_scores)
        n_parts = (singleton_obj_scores.shape[0] > 0) + (nonsingleton_obj_scores.shape[0] > 0)
        loss_obj = (loss_obj_s + loss_obj_ns) / max(1, n_parts)

        return loss_attr, loss_obj

    def _utilization_loss_rank_conditioned(self, singleton_attr_scores,
                                           singleton_obj_scores,
                                           nonsingleton_attr_scores,
                                           nonsingleton_obj_scores):
        """
        Rank-conditioned utilization: within each candidate rank level, push
        for uniform soft usage. Uses sharpened softmax on inclusion scores
        (differentiable).

        For each rank group, extracts the relevant columns from the soft
        probability matrix, normalizes within the group, and computes KL
        from uniform.

        Singletons and non-singletons use different candidate sets and are
        processed independently.
        """
        ce = self.concept_encoder
        tau = self.utilization_tau

        def _rank_cond_one(scores, candidate_ranks_list):
            """Compute rank-conditioned KL for one score matrix (B, C)."""
            if scores.shape[0] == 0:
                return torch.tensor(0.0, device=scores.device)
            device = scores.device
            cand_ranks = torch.tensor(candidate_ranks_list, device=device, dtype=torch.float)
            unique_ranks = cand_ranks.unique()
            soft_probs = F.softmax(scores / tau, dim=-1)  # (B, C)
            avg_usage = soft_probs.mean(dim=0)  # (C,)

            total = 0.0
            n_groups = 0
            for rank_val in unique_ranks:
                cand_mask = (cand_ranks == rank_val)
                n_cands = cand_mask.sum().item()
                if n_cands < 2:
                    continue
                # Extract soft usage for candidates at this rank
                group_usage = avg_usage[cand_mask]  # (n_cands,)
                # Normalize to empirical distribution within group
                emp = group_usage / (group_usage.sum() + 1e-8)
                uni = torch.ones(n_cands, device=device) / n_cands
                total = total + (emp * ((emp + 1e-8).log() - uni.log())).sum()
                n_groups += 1
            return total / max(1, n_groups)

        # Singletons: all candidates have rank = n_attr (attr) or 1 (obj)
        loss_attr_s = _rank_cond_one(
            singleton_attr_scores, ce._singleton_attr_candidate_ranks
        )
        loss_attr_ns = _rank_cond_one(
            nonsingleton_attr_scores, ce._attr_candidate_ranks
        )
        n_parts = (singleton_attr_scores.shape[0] > 0) + (nonsingleton_attr_scores.shape[0] > 0)
        loss_attr = (loss_attr_s + loss_attr_ns) / max(1, n_parts)

        loss_obj_s = _rank_cond_one(
            singleton_obj_scores, ce._singleton_obj_candidate_ranks
        )
        loss_obj_ns = _rank_cond_one(
            nonsingleton_obj_scores, ce._obj_candidate_ranks
        )
        n_parts = (singleton_obj_scores.shape[0] > 0) + (nonsingleton_obj_scores.shape[0] > 0)
        loss_obj = (loss_obj_s + loss_obj_ns) / max(1, n_parts)

        return loss_attr, loss_obj

    def _assignment_sharpness_loss(self, attr_scores, obj_scores):
        """
        Per-image assignment entropy loss: penalizes uniform soft assignments.

        For each image, computes entropy of softmax(scores / tau). Minimizing
        this pushes each image to strongly prefer one memory entry, breaking
        the equidistant equilibrium where all proposals sit at equal distance
        from all entries.

        Works synergistically with batch-level utilization loss:
        - This loss: "each image should pick a clear winner" (low per-image entropy)
        - Utilization: "winners should be spread across all entries" (high batch entropy)

        Returns: (loss_attr, loss_obj) — mean per-image entropy for each.
        """
        tau = self.utilization_tau
        losses = []
        for scores in [attr_scores, obj_scores]:
            if scores.shape[0] == 0:
                losses.append(torch.tensor(0.0, device=scores.device))
                continue
            soft_probs = F.softmax(scores / tau, dim=-1)  # (B, C)
            # Per-image entropy: -sum(p * log(p))
            per_image_entropy = -(soft_probs * (soft_probs + 1e-8).log()).sum(dim=-1)  # (B,)
            losses.append(per_image_entropy.mean())
        return losses[0], losses[1]

    # ================================================================== #
    #  Optimizer
    # ================================================================== #
    def configure_optimizers(self):
        opt_config = self.config["trainer"]["optimizer"]
        base_lr = opt_config["config"]["lr"]
        memory_lr_factor = self.model_config.get("memory_lr_factor", 1.0)

        if memory_lr_factor != 1.0:
            # Separate param groups: memory params get scaled LR
            memory_params = []
            other_params = []
            memory_names = {"obj_memory", "attr_memory"}
            for name, param in self.named_parameters():
                is_memory = any(mn in name for mn in memory_names)
                if is_memory:
                    memory_params.append(param)
                else:
                    other_params.append(param)
            param_groups = [
                {"params": other_params, "lr": base_lr},
                {"params": memory_params, "lr": base_lr * memory_lr_factor},
            ]
        else:
            param_groups = self.parameters()

        if opt_config["type"] == "Adam":
            return torch.optim.Adam(param_groups, lr=base_lr)
        raise ValueError("Unsupported optimizer")

    # ================================================================== #
    #  Epoch-end visualization
    # ================================================================== #
    def on_train_epoch_end(self):
        if self.viz_datapoint is not None:
            fig = self._reconstruction_figure(
                self.viz_datapoint, self.current_epoch
            )
            self.logger.experiment.log({"reconstruction_visualization": fig})
            self.viz_datapoint = None
            plt.close(fig)

        heatmap_tensor, memory_fig_tensor, ranks_tensor = self._log_lattice_and_memory()
        log_dict = {"lattice_inclusion_heatmap": wandb.Image(heatmap_tensor)}
        if memory_fig_tensor is not None:
            log_dict["memory_entries"] = wandb.Image(memory_fig_tensor)
        log_dict["concept_ranks"] = wandb.Image(ranks_tensor)
        self.logger.experiment.log(log_dict)

    # ================================================================== #
    #  Visualization helpers
    # ================================================================== #
    @staticmethod
    def _reconstruction_figure(data, epoch):
        original = data["original_images"]
        recon = data["reconstructed_images"]
        grid_in = torchvision.utils.make_grid(
            original, nrow=original.shape[0], padding=2
        )
        grid_re = torchvision.utils.make_grid(
            recon, nrow=recon.shape[0], padding=2
        )
        fig = plt.figure(figsize=(12, 6))
        plt.suptitle(f"Epoch {epoch} Reconstructions", fontsize=16)
        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(np.clip(grid_in.permute(1, 2, 0).numpy(), 0, 1))
        ax1.set_title("Original"); ax1.axis("off")
        ax2 = plt.subplot(2, 1, 2)
        ax2.imshow(np.clip(grid_re.permute(1, 2, 0).numpy(), 0, 1))
        ax2.set_title("Reconstructed"); ax2.axis("off")
        plt.tight_layout()
        return fig

    @torch.no_grad()
    def _log_lattice_and_memory(self):
        """
        Generate canonical images, resolve through memory, and plot:
          1. Inclusion heatmap (attr + obj)
          2. Memory entry bar charts (attr slots + obj memory)
        """
        self.eval()
        tc = self.config["data"]["train"]["config"]
        img_size = tc.get("image_size", 64)
        if isinstance(img_size, (list, tuple)):
            sx, sy = img_size
        else:
            sx = sy = img_size
        shapes = tc.get("shapes", ["circle", "square"])
        colors = tc.get("colors", ["red", "blue"])
        excluded = set(tc.get("excluded_combinations", []))
        color_map = {
            "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
            "yellow": (255, 255, 0), "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
        }
        center_v = tc.get("center_range", [32, 33])[0]
        size_v = tc.get("size_range", [20, 21])[0]
        tf = transforms.ToTensor()

        def _make(shape, cname):
            img = np.zeros([sx, sy, 3], dtype=np.uint8)
            c = color_map.get(cname, (255, 255, 255))
            ctr = (center_v, center_v)
            if shape == "circle":
                cv2.circle(img, ctr, size_v // 2, c, -1)
            elif shape == "square":
                h = size_v // 2
                cv2.rectangle(img, (ctr[0]-h, ctr[1]-h), (ctr[0]+h, ctr[1]+h), c, -1)
            return tf(img).to(self.device)

        combos = list(itertools.product(colors, shapes))
        imgs = torch.stack([_make(s, c) for c, s in combos])
        labels = [
            f"{c.capitalize()} {s.capitalize()}"
            + (" *" if f"{c} {s}" in excluded else "")
            for c, s in combos
        ]
        N = len(combos)

        R = self.perceptual_encoder(imgs)  # (N, P, E)

        # Encode each singleton
        R_singletons = R.unsqueeze(1)  # (N, 1, P, E)
        enc_out = self.concept_encoder(
            R_singletons, tau=0.01, hard=True, singleton=True
        )
        X_attr = enc_out["resolved_X_attr"]
        X_obj = enc_out["resolved_X_obj"]
        P_attr = enc_out["resolved_P_attr"]
        P_obj = enc_out["resolved_P_obj"]
        attr_idx = enc_out["attr_assign_idx"]
        obj_idx = enc_out["obj_assign_idx"]

        # Build concept groups
        concept_defs = {"C0 (Universal)": list(range(N))}
        ci = 1
        for color in colors:
            concept_defs[f"C{ci} ({color.capitalize()})"] = [
                i for i, (c, s) in enumerate(combos) if c == color
            ]
            ci += 1
        for shape in shapes:
            concept_defs[f"C{ci} ({shape.capitalize()})"] = [
                i for i, (c, s) in enumerate(combos) if s == shape
            ]
            ci += 1
        for i, (c, s) in enumerate(combos):
            lbl = f"{c.capitalize()} {s.capitalize()}"
            if f"{c} {s}" in excluded:
                lbl += " *"
            concept_defs[f"C{ci} ({lbl})"] = [i]
            ci += 1

        concept_labels = list(concept_defs.keys())
        n_concepts = len(concept_labels)

        # Compute concept projectors for multi-image concepts
        P_attr_concepts = []
        P_obj_concepts = []
        for name, idxs in concept_defs.items():
            if len(idxs) == 1:
                P_attr_concepts.append(P_attr[idxs[0]])
                P_obj_concepts.append(P_obj[idxs[0]])
            else:
                # Encode the multi-image subset
                R_sub = R[idxs].unsqueeze(0)  # (1, k, P, E)
                enc_sub = self.concept_encoder(
                    R_sub, tau=0.01, hard=True
                )
                P_attr_concepts.append(enc_sub["resolved_P_attr"].squeeze(0))
                P_obj_concepts.append(enc_sub["resolved_P_obj"].squeeze(0))

        P_attr_concepts = torch.stack(P_attr_concepts)
        P_obj_concepts = torch.stack(P_obj_concepts)

        # Inclusion heatmaps
        attr_inc = np.zeros((N, n_concepts))
        obj_inc = np.zeros((N, n_concepts))
        P_attr_singles = P_attr[:N]
        P_obj_singles = P_obj[:N]

        for i in range(N):
            for j in range(n_concepts):
                attr_inc[i, j] = get_inclusion(
                    P_attr_concepts[j].unsqueeze(0),
                    P_attr_singles[i].unsqueeze(0),
                ).item()
                obj_inc[i, j] = get_inclusion(
                    P_obj_singles[i].unsqueeze(0),
                    P_obj_concepts[j].unsqueeze(0),
                ).item()

        fig_inc, axes = plt.subplots(
            2, 1, figsize=(max(12, n_concepts * 0.8), max(10, N * 1.5))
        )
        sns.heatmap(
            obj_inc, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=concept_labels, yticklabels=labels, ax=axes[0],
        )
        axes[0].set_title("Obj Inclusion: P(singleton_obj ⊆ concept_obj)")
        axes[0].tick_params(axis="x", labelrotation=45)
        axes[0].tick_params(axis="y", labelrotation=0)

        sns.heatmap(
            attr_inc, annot=True, fmt=".2f", cmap="Oranges", vmin=0, vmax=1,
            xticklabels=concept_labels, yticklabels=labels, ax=axes[1],
        )
        axes[1].set_title("Attr Inclusion: P(concept_attr ⊆ singleton_attr)")
        axes[1].tick_params(axis="x", labelrotation=45)
        axes[1].tick_params(axis="y", labelrotation=0)
        plt.tight_layout()
        inc_tensor = self._fig_to_tensor(fig_inc)

        # Concept rank bar chart (attr + obj ranks for all 9 concepts)
        attr_ranks_viz = torch.einsum(
            "bii->b", P_attr_concepts
        ).detach().cpu().numpy()
        obj_ranks_viz = torch.einsum(
            "bii->b", P_obj_concepts
        ).detach().cpu().numpy()

        fig_ranks, ax_ranks = plt.subplots(1, 1, figsize=(max(12, n_concepts * 1.2), 5))
        x_pos = np.arange(n_concepts)
        bar_w = 0.35
        ax_ranks.bar(x_pos - bar_w / 2, attr_ranks_viz, bar_w,
                     label="Attr Rank", color="#e07b39", alpha=0.85)
        ax_ranks.bar(x_pos + bar_w / 2, obj_ranks_viz, bar_w,
                     label="Obj Rank", color="#2b6ca3", alpha=0.85)
        # Annotate values
        for i in range(n_concepts):
            ax_ranks.text(i - bar_w / 2, attr_ranks_viz[i] + 0.05,
                          f"{attr_ranks_viz[i]:.2f}", ha="center", va="bottom",
                          fontsize=7, color="#e07b39")
            ax_ranks.text(i + bar_w / 2, obj_ranks_viz[i] + 0.05,
                          f"{obj_ranks_viz[i]:.2f}", ha="center", va="bottom",
                          fontsize=7, color="#2b6ca3")
        ax_ranks.set_xticks(x_pos)
        ax_ranks.set_xticklabels(concept_labels, rotation=45, ha="right", fontsize=8)
        ax_ranks.set_ylabel("Rank (Tr(P))")
        ax_ranks.set_title(f"Concept Ranks — Epoch {self.current_epoch}")
        ax_ranks.legend()
        ax_ranks.grid(True, alpha=0.3, axis="y")
        ax_ranks.set_ylim(0, max(float(self.n_attr), float(self.n_obj)) + 0.5)
        plt.tight_layout()
        ranks_tensor = self._fig_to_tensor(fig_ranks)

        # Memory bar charts
        n_charts = self.n_attr + 1  # attr slots + obj memory
        fig_mem, axes_mem = plt.subplots(1, n_charts, figsize=(6 * n_charts, 4))
        if n_charts == 1:
            axes_mem = [axes_mem]

        for s in range(self.n_attr):
            M = self.concept_encoder.attr_memory[s].detach().cpu().numpy()
            ax = axes_mem[s]
            K, d = M.shape
            x = np.arange(d)
            w = 0.8 / K
            for k_idx in range(K):
                ax.bar(x + k_idx * w, M[k_idx], width=w, alpha=0.8,
                       label=f"Entry {k_idx}")
            ax.set_title(f"Attr Slot {s} ({K} fillers, {d} dims)")
            ax.set_xlabel("Dim"); ax.set_ylabel("Value")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Obj memory — each entry is (n_obj, ambient_dim), show norms per entry
        M_obj = self.concept_encoder.obj_memory.detach().cpu().numpy()
        ax_obj = axes_mem[self.n_attr]
        n_o, n_basis, D = M_obj.shape
        # Show per-entry basis vector norms as a grouped bar chart
        x = np.arange(n_basis)
        w = 0.8 / n_o
        for k_idx in range(n_o):
            norms = np.linalg.norm(M_obj[k_idx], axis=-1)  # (n_basis,)
            ax_obj.bar(x + k_idx * w, norms, width=w, alpha=0.8,
                       label=f"Obj {k_idx}")
        ax_obj.set_title(f"Obj Memory ({n_o} subspaces, {n_basis} basis vecs, {D} dims)")
        ax_obj.set_xlabel("Basis vector"); ax_obj.set_ylabel("Norm")
        ax_obj.legend(fontsize=7)
        ax_obj.grid(True, alpha=0.3)

        # Show assignments
        fig_mem.suptitle(
            "Attr assigns: " + ", ".join(
                f"{labels[i]}→{attr_idx[i].item()}" for i in range(N)
            ) + "  |  Obj assigns: " + ", ".join(
                f"{labels[i]}→{obj_idx[i].item()}" for i in range(N)
            ),
            fontsize=7,
        )
        plt.tight_layout()
        mem_tensor = self._fig_to_tensor(fig_mem)

        self.train()
        return inc_tensor, mem_tensor, ranks_tensor

    @staticmethod
    def _fig_to_tensor(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(arr).permute(2, 0, 1)
