class VQSubspaceConceptLattice(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config["model"]["config"]

        self.embed_dim = self.model_config["embed_dim"]
        self.attr_ambient_dim = self.model_config["attr_ambient_dim"]
        self.obj_ambient_dim = self.model_config["obj_ambient_dim"]
        self.attr_structure = self.model_config["attr_structure"]
        self.n_obj = self.model_config["n_obj"]

        self.lbd = self.model_config["lbd"]
        self.image_size = self.model_config["image_size"]
        self.image_channels = self.model_config["image_channels"]

        self.num_pairwise_iter = self.model_config["num_pairwise_iter"]
        self.loss_weights = self.model_config["loss_weights"]

        if self.model_config["perceptual_encoder"]["type"] == "ViTEncoder":
            perceptual_encoder_config = self.model_config["perceptual_encoder"]["config"]
            perceptual_encoder_config.update({
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
                "image_channels": self.image_channels,
            })
            self.perceptual_encoder = ViTEncoder(perceptual_encoder_config)

        if self.model_config["concept_encoder"]["type"] == "VQConceptEncoder":
            concept_encoder_config = self.model_config["concept_encoder"]["config"]
            concept_encoder_config.update({
                "embed_dim": self.embed_dim,
                "attr_ambient_dim": self.attr_ambient_dim,
                "obj_ambient_dim": self.obj_ambient_dim,
                "attr_structure": self.attr_structure,
                "n_obj": self.n_obj,
                "lbd": self.lbd,
            })
            self.concept_encoder = VQConceptEncoder(concept_encoder_config)

        if self.model_config["decoder"]["type"] == "VQDecoder":
            decoder_config = self.model_config["decoder"]["config"]
            decoder_config.update({
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
                "image_channels": self.image_channels,
                "attr_ambient_dim": self.attr_ambient_dim
            })
            self.decoder = VQDecoder(decoder_config)

        self.viz_datapoint = None

    def forward(self, x):
        images = x["images"] # (B, 3, H, W)
        representations = self.perceptual_encoder(images) # (B, num_patches, embed_dim)
        B, num_patches, _ = representations.shape

        X_attr_singleton, X_obj_singleton, attr_commitment_loss_singleton = self.concept_encoder(representations.view(B, 1, num_patches, self.embed_dim))

        combos = list(itertools.combinations(range(B), 2))
        combos_tensor = torch.tensor(combos, device=self.device) # (len(combos), 2)
        R_subset = representations[combos_tensor] # (len(combos), 2, num_patches, embed_dim)
        X_attr, X_obj, attr_commitment_loss = self.concept_encoder(R_subset)

        total_attr_commitment_loss = attr_commitment_loss_singleton + attr_commitment_loss
        total_attr_commitment_loss /= 2

        P_attr_singleton = torch.diag_embed(X_attr_singleton)
        reconstructed_images = self.decoder(P_attr_singleton)

        return {
            "images": images,
            "reconstructed_images": reconstructed_images,
            "X_attr": X_attr,
            "X_obj": X_obj,
            "combo_indices": combos_tensor,
            "X_attr_singleton": X_attr_singleton,
            "X_obj_singleton": X_obj_singleton,
            "attr_commitment_loss": total_attr_commitment_loss
        }

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        images = outputs["images"] # (B, 3, H, W)
        reconstructed_images = outputs["reconstructed_images"] # (B, 3, H, W)
        B = images.shape[0]

        X_attr = outputs["X_attr"]
        X_obj = outputs["X_obj"]
        X_attr_singleton = outputs["X_attr_singleton"]
        X_obj_singleton = outputs["X_obj_singleton"]
        combo_indices = outputs["combo_indices"]
        attr_commitment_loss = outputs["attr_commitment_loss"]

        if self.viz_datapoint is None:
            num_samples = min(4, B)
            self.viz_datapoint = {
                "original_images": images[:num_samples].detach().cpu(),
                "reconstructed_images": reconstructed_images[:num_samples].detach().cpu(),
            }

        # LOSS COMPUTATIONS

        ## Reconstruction Loss
        reconstruction_loss = F.mse_loss(reconstructed_images, images, reduction="sum") / B

        ## Intersection Loss
        total_attr_intersection_loss = 0
        ## Union Loss
        total_obj_union_loss = 0
        ## Closure Loss
        total_obj_closure_loss = 0

        num_abstract_combos = combo_indices.shape[0]

        for idx in range(num_abstract_combos):
            combo = combo_indices[idx]
            combo_X_attr = X_attr[i] # (attr_ambient_dim,)
            combo_X_obj = X_obj[i]   # (n_obj,)

            combo_singleton_X_attr = X_attr_singleton[combo] # (2, attr_ambient_dim)
            combo_singleton_X_obj = X_obj_singleton[combo] # (2, n_obj)

            target_attr_intersection = torch.min(combo_singleton_X_attr, dim=0).values # (attr_ambient_dim,)
            combo_attr_intersection_loss = F.mse_loss(combo_X_attr, target_attr_intersection)

            target_obj_union = torch.max(combo_singleton_X_obj, dim=0).values # (n_obj,)
            union_penalty = torch.relu(target_obj_union - combo_X_obj)
            combo_obj_union_loss = torch.mean(union_penalty ** 2)

            missing_attrs = torch.relu(combo_X_attr.unsqueeze(0) - X_attr_singleton) # (B, attr_ambient_dim)
            missing_sum = missing_attrs.sum(dim=1) # (B,)
            target_extent = (missing_sum == 0).float() # (B,)
            combo_obj_closure_loss = F.mse_loss(combo_X_obj, target_extent)

            total_attr_intersection_loss += combo_attr_intersection_loss
            total_obj_union_loss += combo_obj_union_loss
            total_obj_closure_loss += combo_obj_closure_loss

        total_attr_intersection_loss /= num_abstract_combos
        total_obj_union_loss /= num_abstract_combos
        total_obj_closure_loss /= num_abstract_combos
        
        total_loss = (
            self.loss_weights["reconstruction_loss"] * reconstruction_loss +
            self.loss_weights["attr_intersection_loss"] * total_attr_intersection_loss +
            self.loss_weights["obj_union_loss"] * total_obj_union_loss +
            self.loss_weights["obj_closure_loss"] * total_obj_closure_loss + 
            self.loss_weights["attr_commitment_loss"] * attr_commitment_loss
        )

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "attr_intersection_loss": total_attr_intersection_loss,
            "obj_union_loss": total_obj_union_loss,
            "obj_closure_loss": total_obj_closure_loss,
            "attr_commitment_loss": attr_commitment_loss,
            "total_loss": total_loss
        }

        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)

        return total_loss