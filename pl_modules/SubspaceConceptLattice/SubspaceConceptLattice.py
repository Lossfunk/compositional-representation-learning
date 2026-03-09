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
from .ConceptEncoder import ConceptEncoder
from .Decoder import ViTDecoder
from .concept_utils import get_inclusion, ridge_projector

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

        self.max_samples_per_cardinality = self.model_config["max_samples_per_cardinality"]
        self.intersection_power_steps = self.model_config["intersection_power_steps"]
        self.intersection_consistency_only_pairs = self.model_config["intersection_consistency_only_pairs"]
        self.maximize_singleton_attr_rank = self.model_config["maximize_singleton_attr_rank"]
        self.loss_weights = self.model_config["loss_weights"]

        if self.model_config["perceptual_encoder"]["type"] == "ViTEncoder":
            perceptual_encoder_config = self.model_config["perceptual_encoder"]["config"]
            perceptual_encoder_config.update({
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
                "image_channels": self.image_channels,
            })
            self.perceptual_encoder = ViTEncoder(perceptual_encoder_config)
        
        if self.model_config["concept_encoder"]["type"] == "ConceptEncoder":
            concept_encoder_config = self.model_config["concept_encoder"]["config"]
            concept_encoder_config.update({
                "embed_dim": self.embed_dim,
                "ambient_dim": self.ambient_dim,
                "n_attr": self.n_attr,
                "n_obj": self.n_obj,
                "lbd": self.lbd,
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
        all_combo_indices = []

        for k in range(1, B + 1):
            combos = list(itertools.combinations(range(B), k))
            if len(combos) > self.max_samples_per_cardinality and k > 1:
                combos = random.sample(combos, self.max_samples_per_cardinality)
            
            combos_tensor = torch.tensor(combos, device=self.device) # (len(combos), k)
            R_subset = representations[combos_tensor] # (len(combos), k, num_patches, embed_dim)
            X_attr, X_obj, P_attr, P_obj = self.concept_encoder(R_subset)

            all_X_attr.append(X_attr) # (len(combos), n_attr, d_ambient)
            all_X_obj.append(X_obj) # (len(combos), n_obj, d_ambient)
            all_P_attr.append(P_attr) # (len(combos), d_ambient, d_ambient)
            all_P_obj.append(P_obj) # (len(combos), d_ambient, d_ambient)
            all_cardinalities.append(torch.tensor([k] * len(combos), device=self.device)) # (len(combos),)
            all_combo_indices.extend(combos_tensor) # (len(combos), k)

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
            "combo_indices": all_combo_indices,
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
        combo_indices = outputs["combo_indices"]
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

        for cardinality in range(2, B + 1):
            cardinality_mask = (cardinalities_tensor == cardinality)
            cardinality_combo_indices = torch.stack([combo_indices[i] for i in torch.where(cardinality_mask)[0]], dim=0).to(self.device) # (num_combos, k)
            num_combos = cardinality_combo_indices.shape[0]

            cardinality_P_attr = P_attr_tensor[cardinality_mask] # (num_combos, ambient_dim, ambient_dim)
            cardinality_P_obj = P_obj_tensor[cardinality_mask] # (num_combos, ambient_dim, ambient_dim)

            combo_singletons_P_attr = P_attr_singletons[cardinality_combo_indices] # (num_combos, k, ambient_dim, ambient_dim)
            combo_singletons_P_obj = P_obj_singletons[cardinality_combo_indices] # (num_combos, k, ambient_dim, ambient_dim)
            
            combo_singletons_X_obj = X_obj_singletons[cardinality_combo_indices] # (num_combos, k, n_obj, ambient_dim)
            n_obj = combo_singletons_X_obj.shape[2]
            X_obj_union_basis = combo_singletons_X_obj.view(num_combos, cardinality * n_obj, self.ambient_dim) # (num_combos, k * n_obj, ambient_dim)
            P_obj_union = ridge_projector(X_obj_union_basis, lbd=self.lbd) # (num_combos, ambient_dim, ambient_dim)
            total_union_consistency_loss += F.mse_loss(cardinality_P_obj, P_obj_union, reduction="sum")

            if self.intersection_consistency_only_pairs:
                if cardinality == 2:
                    Pi = combo_singletons_P_attr[:, 0] # (num_combos, ambient_dim, ambient_dim)
                    Pj = combo_singletons_P_attr[:, 1] # (num_combos, ambient_dim, ambient_dim)
                    combo_singletons_P_attr_inter = 0.5 * (Pi @ Pj + Pj @ Pi) # (num_combos, ambient_dim, ambient_dim)
                    total_intersection_consistency_loss += F.mse_loss(cardinality_P_attr, combo_singletons_P_attr_inter, reduction="sum")

                    Xi = combo_singletons_X_obj[:, 0] # (num_combos, n_obj, ambient_dim)
                    Xj = combo_singletons_X_obj[:, 1] # (num_combos, n_obj, ambient_dim)
                    Xi_union_Xj = torch.cat([Xi, Xj], dim=1) # (num_combos, 2 * n_obj, ambient_dim)
                    Xi_union_Xj_P_attr = ridge_projector(Xi_union_Xj, lbd=self.lbd) # (num_combos, ambient_dim, ambient_dim)
                    target_inter_rank = torch.einsum("bii->b", Pi) + torch.einsum("bii->b", Pj) - torch.einsum("bii->b", Xi_union_Xj_P_attr) # (num_combos,)
                    actual_inter_rank = torch.einsum("bii->b", cardinality_P_attr) # (num_combos,)
                    total_modular_subspace_loss += F.mse_loss(actual_inter_rank, target_inter_rank, reduction="sum")
            else:
                combo_singletons_P_attr_avg = torch.mean(combo_singletons_P_attr, dim=1) # (num_combos, ambient_dim, ambient_dim)
                combo_singletons_P_attr_inter = torch.linalg.matrix_power(combo_singletons_P_attr_avg, self.intersection_power_steps) # (num_combos, ambient_dim, ambient_dim)
                total_intersection_consistency_loss += F.mse_loss(cardinality_P_attr, combo_singletons_P_attr_inter, reduction="sum")

            P_attr_comb_exp = cardinality_P_attr.unsqueeze(1).expand(-1, cardinality, -1, -1) # (num_combos, k, ambient_dim, ambient_dim)
            P_obj_comb_exp = cardinality_P_obj.unsqueeze(1).expand(-1, cardinality, -1, -1) # (num_combos, k, ambient_dim, ambient_dim)

            P_attr_c_flat = P_attr_comb_exp.reshape(-1, self.ambient_dim, self.ambient_dim) # (num_combos * k, ambient_dim, ambient_dim)
            P_attr_s_flat = combo_singletons_P_attr.reshape(-1, self.ambient_dim, self.ambient_dim) # (num_combos * k, ambient_dim, ambient_dim)
            P_obj_c_flat = P_obj_comb_exp.reshape(-1, self.ambient_dim, self.ambient_dim) # (num_combos * k, ambient_dim, ambient_dim)
            P_obj_s_flat = combo_singletons_P_obj.reshape(-1, self.ambient_dim, self.ambient_dim) # (num_combos * k, ambient_dim, ambient_dim)

            galois_attr_inclusion = get_inclusion(P_sub=P_attr_c_flat, P_super=P_attr_s_flat) # (num_combos * k,)
            galois_obj_inclusion = get_inclusion(P_sub=P_obj_s_flat, P_super=P_obj_c_flat) # (num_combos * k,)

            attr_target = torch.ones_like(galois_attr_inclusion) # (num_combos * k,)
            obj_target = torch.ones_like(galois_obj_inclusion) # (num_combos * k,)

            total_galois_attr_loss += F.binary_cross_entropy(galois_attr_inclusion, attr_target, reduction="sum")
            total_galois_obj_loss += F.binary_cross_entropy(galois_obj_inclusion, obj_target, reduction="sum")
            total_comparisons += num_combos * cardinality

        union_consistency_loss = total_union_consistency_loss / max(1, (B_total - B))
        if self.intersection_consistency_only_pairs:
            intersection_consistency_loss = total_intersection_consistency_loss / 10
            modular_subspace_loss = total_modular_subspace_loss / 10
        else:
            intersection_consistency_loss = total_intersection_consistency_loss / max(1, (B_total - B))
            modular_subspace_loss = total_modular_subspace_loss / max(1, (B_total - B))

        galois_attr_loss = total_galois_attr_loss / max(1, total_comparisons)
        galois_obj_loss = total_galois_obj_loss / max(1, total_comparisons)

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

        total_loss = (
            self.loss_weights["reconstruction_loss"] * reconstruction_loss +
            self.loss_weights["singleton_obj_rank_loss"] * singleton_obj_rank_loss +
            self.loss_weights["max_singleton_attr_rank_loss"] * max_singleton_attr_rank_loss +
            self.loss_weights["modular_subspace_loss"] * modular_subspace_loss +
            self.loss_weights["attr_orthogonality_loss"] * attr_orthogonality_loss +
            self.loss_weights["concept_similarity_loss"] * concept_similarity_loss +
            self.loss_weights["union_consistency_loss"] * union_consistency_loss +
            self.loss_weights["intersection_consistency_loss"] * intersection_consistency_loss +
            self.loss_weights["galois_attr_loss"] * galois_attr_loss +
            self.loss_weights["galois_obj_loss"] * galois_obj_loss +
            self.loss_weights["loss_obj_card_prop"] * loss_obj_card_prop +
            self.loss_weights["loss_attr_card_inv_prop"] * loss_attr_card_inv_prop +
            self.loss_weights["loss_attr_obj_inv_prop"] * loss_attr_obj_inv_prop +
            self.loss_weights["repulsion_loss_obj"] * repulsion_loss_obj +
            self.loss_weights["repulsion_loss_attr"] * repulsion_loss_attr
        )

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "singleton_obj_rank_loss": singleton_obj_rank_loss,
            "max_singleton_attr_rank_loss": max_singleton_attr_rank_loss,
            "modular_subspace_loss": modular_subspace_loss,
            "attr_orthogonality_loss": attr_orthogonality_loss,
            "concept_similarity_loss": concept_similarity_loss,
            "union_consistency_loss": union_consistency_loss,
            "intersection_consistency_loss": intersection_consistency_loss,
            "galois_attr_loss": galois_attr_loss,
            "galois_obj_loss": galois_obj_loss,
            "loss_obj_card_prop": loss_obj_card_prop,
            "loss_attr_card_inv_prop": loss_attr_card_inv_prop,
            "loss_attr_obj_inv_prop": loss_attr_obj_inv_prop,
            "repulsion_loss_obj": repulsion_loss_obj,
            "repulsion_loss_attr": repulsion_loss_attr,
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
        gram = torch.bmm(X, X.transpose(1, 2)) # (B, N, N)
        
        # Normalize to get cosine similarities
        norms = torch.norm(X, dim=-1, keepdim=True) + 1e-8
        gram_norm = gram / torch.bmm(norms, norms.transpose(1, 2))
        
        # Mask out the diagonal
        mask = ~torch.eye(N, dtype=torch.bool, device=X.device).expand(B, N, N)
        
        # Penalize off-diagonal correlations
        return (gram_norm[mask] ** 2).mean()

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

        # Generate all valid combinations dynamically
        combinations = list(itertools.product(colors, shapes))
        
        images = []
        singleton_labels = []
        for c, s in combinations:
            images.append(make_ideal_img(s, c))
            singleton_labels.append(f"{c.capitalize()} {s.capitalize()}")
        
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
        
        for name, indices in concept_defs.items():
            subset = R[indices].unsqueeze(0) # (1, subset_size, num_patches, embed_dim)
            _, _, P_attr, P_obj = self.concept_encoder(subset)
            P_attr_concepts.append(P_attr.squeeze(0)) # (ambient_dim, ambient_dim)
            P_obj_concepts.append(P_obj.squeeze(0))   # (ambient_dim, ambient_dim)
            
        P_attr_concepts = torch.stack(P_attr_concepts) # (num_concepts, D, D)
        P_obj_concepts = torch.stack(P_obj_concepts)   # (num_concepts, D, D)
        
        # Extract projectors for the singletons (last `num_singletons` concepts)
        P_attr_singletons = P_attr_concepts[-num_singletons:] # (num_singletons, D, D)
        P_obj_singletons = P_obj_concepts[-num_singletons:]   # (num_singletons, D, D)
        
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
        
        self.train() # restore training mode
        return heatmap_tensor

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

        inclusion_fig = self.log_concept_lattice_inclusion()
        self.logger.experiment.log({
            "lattice_inclusion_heatmap": wandb.Image(inclusion_fig),
        })
        plt.close(inclusion_fig)

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