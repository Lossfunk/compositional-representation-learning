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
        self.loss_weights = self.model_config["loss_weights"]

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

        all_X_attr = []
        all_X_obj = []
        all_P_attr = []
        all_P_obj = []
        all_cardinalities = []
        all_combination_indices = []
        all_attr_assign_idx = []
        all_obj_assign_idx = []
        all_attr_log_probs = []
        all_obj_log_probs = []
        singleton_attr_assign_idx = []
        singleton_obj_assign_idx = []
        total_commitment_loss = 0.0
        n_groups = 0

        for k in range(1, B + 1):
            combination_indices = list(itertools.combinations(range(B), k))
            if len(combination_indices) > self.max_combinations_per_cardinality and k > 1:
                combination_indices = random.sample(
                    combination_indices, self.max_combinations_per_cardinality
                )

            combo_tensor = torch.tensor(combination_indices, device=self.device)
            R_subset = representations[combo_tensor]  # (num_combos, k, P, E)

            is_singleton = (k == 1)
            (X_attr, X_obj, P_attr, P_obj,
             attr_idx, obj_idx, attr_lp, obj_lp,
             commit) = self.concept_encoder(
                R_subset, tau=tau, hard=hard, singleton=is_singleton
            )

            all_X_attr.append(X_attr)
            all_X_obj.append(X_obj)
            all_P_attr.append(P_attr)
            all_P_obj.append(P_obj)
            all_cardinalities.append(
                torch.full((len(combination_indices),), k, device=self.device)
            )
            all_combination_indices.extend(combo_tensor)
            all_attr_assign_idx.append(attr_idx)
            all_obj_assign_idx.append(obj_idx)
            all_attr_log_probs.append(attr_lp)
            all_obj_log_probs.append(obj_lp)
            if is_singleton:
                singleton_attr_assign_idx.append(attr_idx)
                singleton_obj_assign_idx.append(obj_idx)
            total_commitment_loss += commit
            n_groups += 1

        X_attr_tensor = torch.cat(all_X_attr, dim=0)
        X_obj_tensor = torch.cat(all_X_obj, dim=0)
        P_attr_tensor = torch.cat(all_P_attr, dim=0)
        P_obj_tensor = torch.cat(all_P_obj, dim=0)
        cardinalities = torch.cat(all_cardinalities, dim=0)
        attr_assign_idx = torch.cat(all_attr_assign_idx, dim=0)
        obj_assign_idx = torch.cat(all_obj_assign_idx, dim=0)
        attr_log_probs = torch.cat(all_attr_log_probs, dim=0)
        obj_log_probs = torch.cat(all_obj_log_probs, dim=0)
        singleton_attr_idx = torch.cat(singleton_attr_assign_idx, dim=0)
        singleton_obj_idx = torch.cat(singleton_obj_assign_idx, dim=0)
        commitment_loss = total_commitment_loss / max(1, n_groups)

        singletons_mask = (cardinalities == 1)
        X_attr_singletons = X_attr_tensor[singletons_mask]
        X_attr_dec = self.concept_encoder.attr_dec(X_attr_singletons)
        reconstructed = self.decoder(X_attr_dec)

        return {
            "images": images,
            "reconstructed_images": reconstructed,
            "representations": representations,
            "X_attr_tensor": X_attr_tensor,
            "X_obj_tensor": X_obj_tensor,
            "P_attr_tensor": P_attr_tensor,
            "P_obj_tensor": P_obj_tensor,
            "cardinalities": cardinalities,
            "combination_indices": all_combination_indices,
            "singletons_mask": singletons_mask,
            "attr_assign_idx": attr_assign_idx,
            "obj_assign_idx": obj_assign_idx,
            "singleton_attr_assign_idx": singleton_attr_idx,
            "singleton_obj_assign_idx": singleton_obj_idx,
            "attr_log_probs": attr_log_probs,
            "obj_log_probs": obj_log_probs,
            "commitment_loss": commitment_loss,
        }

    # ================================================================== #
    #  Training step
    # ================================================================== #
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        images = outputs["images"]
        reconstructed = outputs["reconstructed_images"]
        B = images.shape[0]

        X_attr = outputs["X_attr_tensor"]
        X_obj = outputs["X_obj_tensor"]
        P_attr = outputs["P_attr_tensor"]
        P_obj = outputs["P_obj_tensor"]
        cardinalities = outputs["cardinalities"]
        combination_indices = outputs["combination_indices"]
        singletons_mask = outputs["singletons_mask"]
        commitment_loss = outputs["commitment_loss"]
        B_total = X_attr.shape[0]

        P_attr_singletons = P_attr[singletons_mask]
        P_obj_singletons = P_obj[singletons_mask]
        X_attr_singletons = X_attr[singletons_mask]
        X_obj_singletons = X_obj[singletons_mask]

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

        # ── Singleton losses ────────────────────────────────────────────
        singleton_attr_ranks = torch.einsum("bii->b", P_attr_singletons)
        max_singleton_attr_rank_loss = -torch.mean(singleton_attr_ranks)

        # ── Repulsion losses (within batch singletons) ──────────────────
        repulsion_loss_attr = self._compute_soft_repulsion(P_attr_singletons)
        repulsion_loss_obj = self._compute_soft_repulsion(P_obj_singletons)

        # ── Combination losses (Galois, IC, UC, sink) ───────────────────
        total_galois_attr_loss = 0.0
        total_galois_obj_loss = 0.0
        total_intersection_consistency_loss = 0.0
        total_union_consistency_loss = 0.0
        total_attr_sink_loss = 0.0
        total_comparisons = 0

        # Get obj ranks for all concepts
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
            card_X_attr = X_attr[card_mask]

            # ── Attr sink loss (obj_rank-weighted) ──────────────────────
            card_obj_ranks = obj_ranks[card_mask]
            # Normalize obj_rank to [0, 1] range for weighting
            max_obj_rank = float(self.n_obj)
            sink_weight = (card_obj_ranks / max_obj_rank).clamp(0, 1)
            card_attr_ranks = attr_ranks[card_mask]
            total_attr_sink_loss += torch.sum((card_attr_ranks * sink_weight) ** 2)

            # ── Galois losses ───────────────────────────────────────────
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

            # ── Intersection consistency (attr) — power iteration ───────
            combo_P_attr_singletons_avg = combo_P_attr_singletons.mean(dim=1)
            P_attr_inter = torch.linalg.matrix_power(combo_P_attr_singletons_avg, 5)
            total_intersection_consistency_loss += F.mse_loss(
                card_P_attr, P_attr_inter, reduction="sum"
            )

            # ── Union consistency (obj) ─────────────────────────────────
            combo_X_obj_singletons = X_obj_singletons[card_combo_indices]
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

        # ── Rank proportionality (attr inv prop to obj rank) ────────────
        loss_attr_obj_inv_prop = self._proportionality_loss(
            attr_ranks, obj_ranks, inverse=True
        )

        # ── VQ losses ───────────────────────────────────────────────────
        assignment_entropy_loss = self._assignment_entropy_loss(
            outputs["attr_log_probs"], outputs["obj_log_probs"]
        )
        utilization_loss = self._utilization_loss(
            outputs["singleton_attr_assign_idx"], outputs["singleton_obj_assign_idx"]
        )

        # ── Total loss ──────────────────────────────────────────────────
        w = self.loss_weights
        total_loss = (
            w["reconstruction_loss"] * reconstruction_loss
            + w["commitment_loss"] * commitment_loss
            + w["memory_slot_orthogonality_loss"] * memory_slot_orthogonality_loss
            + w["memory_obj_rank_loss"] * memory_obj_rank_loss
            + w["memory_obj_orthogonality_loss"] * memory_obj_orthogonality_loss
            + w["max_singleton_attr_rank_loss"] * max_singleton_attr_rank_loss
            + w["galois_attr_loss"] * galois_attr_loss
            + w["galois_obj_loss"] * galois_obj_loss
            + w["intersection_consistency_loss"] * intersection_consistency_loss
            + w["union_consistency_loss"] * union_consistency_loss
            + w["attr_sink_loss"] * attr_sink_loss
            + w["loss_attr_obj_inv_prop"] * loss_attr_obj_inv_prop
            + w["repulsion_loss_attr"] * repulsion_loss_attr
            + w["repulsion_loss_obj"] * repulsion_loss_obj
            + w.get("assignment_entropy_loss", 0) * assignment_entropy_loss
            + w.get("utilization_loss", 0) * utilization_loss
        )

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": commitment_loss,
            "memory_slot_orthogonality_loss": memory_slot_orthogonality_loss,
            "memory_obj_rank_loss": memory_obj_rank_loss,
            "memory_obj_orthogonality_loss": memory_obj_orthogonality_loss,
            "max_singleton_attr_rank_loss": max_singleton_attr_rank_loss,
            "galois_attr_loss": galois_attr_loss,
            "galois_obj_loss": galois_obj_loss,
            "intersection_consistency_loss": intersection_consistency_loss,
            "union_consistency_loss": union_consistency_loss,
            "attr_sink_loss": attr_sink_loss,
            "loss_attr_obj_inv_prop": loss_attr_obj_inv_prop,
            "repulsion_loss_attr": repulsion_loss_attr,
            "repulsion_loss_obj": repulsion_loss_obj,
            "assignment_entropy_loss": assignment_entropy_loss,
            "utilization_loss": utilization_loss,
            "total_loss": total_loss,
            "tau": self.get_tau(),
        }

        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)

        # Log per-cardinality ranks
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
        """Each object memory vector should span a rank-1 subspace."""
        M = self.concept_encoder.obj_memory  # (n_obj, D)
        loss = 0.0
        for i in range(self.n_obj):
            v = M[i]  # (D,)
            norm_sq = (v * v).sum()
            rank = norm_sq / (norm_sq + self.lbd)
            loss += F.mse_loss(rank, torch.ones(1, device=M.device).squeeze())
        return loss / self.n_obj

    def _memory_obj_orthogonality_loss(self):
        """Object memory vectors should be orthogonal to each other."""
        M = self.concept_encoder.obj_memory  # (n_obj, D)
        if self.n_obj < 2:
            return torch.tensor(0.0, device=M.device)
        M_norm = F.normalize(M, dim=-1)
        sim = M_norm @ M_norm.t()  # (n_obj, n_obj)
        mask = ~torch.eye(self.n_obj, dtype=torch.bool, device=M.device)
        return (sim[mask] ** 2).mean()

    # ================================================================== #
    #  Batch-level losses
    # ================================================================== #
    def _compute_soft_repulsion(self, P_singletons):
        """Minimize pairwise overlap of singleton projectors."""
        B = P_singletons.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=P_singletons.device)
        P_vec = P_singletons.reshape(B, -1)
        overlap = P_vec @ P_vec.t()
        mask = ~torch.eye(B, dtype=torch.bool, device=P_singletons.device)
        return overlap[mask].mean()

    def _proportionality_loss(self, x, y, inverse=False, eps=1e-8):
        """Pearson correlation-based proportionality loss."""
        x_c = x - x.mean()
        y_c = y - y.mean()
        covar = (x_c * y_c).sum()
        var_x = (x_c ** 2).sum()
        var_y = (y_c ** 2).sum()
        corr = covar / (torch.sqrt(var_x * var_y) + eps)
        if inverse:
            return 1.0 + corr
        return 1.0 - corr

    # ================================================================== #
    #  VQ losses
    # ================================================================== #
    def _assignment_entropy_loss(self, attr_log_probs, obj_log_probs):
        """Encourage peaked (confident) assignments."""
        return -(attr_log_probs.mean() + obj_log_probs.mean()) / 2.0

    def _utilization_loss(self, attr_assign_idx, obj_assign_idx):
        """Encourage uniform usage of singleton memory candidates (no zeros, no unions)."""
        loss = 0.0
        # Attr candidates (singleton: fillers only, no zero)
        C_attr = self.concept_encoder.n_singleton_attr_candidates
        counts = torch.bincount(attr_assign_idx, minlength=C_attr).float() + 1e-6
        emp = counts / counts.sum()
        uni = torch.ones(C_attr, device=counts.device) / C_attr
        loss += (emp * (emp.log() - uni.log())).sum()
        # Obj candidates (singleton: individual entries, no unions)
        C_obj = self.concept_encoder.n_singleton_obj_candidates
        counts = torch.bincount(obj_assign_idx, minlength=C_obj).float() + 1e-6
        emp = counts / counts.sum()
        uni = torch.ones(C_obj, device=counts.device) / C_obj
        loss += (emp * (emp.log() - uni.log())).sum()
        return loss / 2.0

    # ================================================================== #
    #  Optimizer
    # ================================================================== #
    def configure_optimizers(self):
        if self.config["trainer"]["optimizer"]["type"] == "Adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.config["trainer"]["optimizer"]["config"]["lr"],
            )
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

        heatmap_tensor, memory_fig_tensor = self._log_lattice_and_memory()
        log_dict = {"lattice_inclusion_heatmap": wandb.Image(heatmap_tensor)}
        if memory_fig_tensor is not None:
            log_dict["memory_entries"] = wandb.Image(memory_fig_tensor)
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
        (X_attr, X_obj, P_attr, P_obj,
         attr_idx, obj_idx, _, _, _) = self.concept_encoder(
            R_singletons, tau=0.01, hard=True, singleton=True
        )

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
                (_, _, Pa, Po, _, _, _, _, _) = self.concept_encoder(
                    R_sub, tau=0.01, hard=True
                )
                P_attr_concepts.append(Pa.squeeze(0))
                P_obj_concepts.append(Po.squeeze(0))

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

        # Obj memory
        M_obj = self.concept_encoder.obj_memory.detach().cpu().numpy()
        ax_obj = axes_mem[self.n_attr]
        n_o, D = M_obj.shape
        x = np.arange(D)
        w = 0.8 / n_o
        for k_idx in range(n_o):
            ax_obj.bar(x + k_idx * w, M_obj[k_idx], width=w, alpha=0.8,
                       label=f"Obj {k_idx}")
        ax_obj.set_title(f"Obj Memory ({n_o} objects, {D} dims)")
        ax_obj.set_xlabel("Dim"); ax_obj.set_ylabel("Value")
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
        return inc_tensor, mem_tensor

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
