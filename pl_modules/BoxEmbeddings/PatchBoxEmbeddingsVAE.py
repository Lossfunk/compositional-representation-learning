import torch
from torch import nn
import lightning as L
import torch.nn.functional as F
import numpy as np
import wandb
from matplotlib import pyplot as plt
import lpips
from pytorch_msssim import SSIM

from box_embeddings.parameterizations.box_tensor import (
    BoxTensor,
)
from box_embeddings.modules.volume import Volume
from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.regularization import (
    L2SideBoxRegularizer,
)

from .utils import create_reconstruction_visualization
from models import VanillaVAE


class PatchBoxEmbeddingsVAE(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config["model"]["config"]["embed_dim"]
        self.box_embed_dim = self.embed_dim * 2
        self.hidden_dims = config["model"]["config"]["hidden_dims"]
        self.grid_size = config["model"]["config"]["grid_size"]
        self.image_size = config["data"]["train"]["config"]["image_size"]
        self.gumbel_temp = config["model"]["config"]["gumbel_temp"]
        self.min_side_length = config["model"]["config"]["min_side_length"]
        self.crop_objects = config["model"]["config"].get("crop_objects", False)

        self.loss_weights = config["model"]["config"]["loss_weights"]

        self.vae = VanillaVAE(self.embed_dim, self.hidden_dims)
        self.box_volume = Volume(
            volume_temperature=self.gumbel_temp,
            log_scale=True,
        )
        self.box_intersection = Intersection(intersection_temperature=self.gumbel_temp)
        self.box_volume_regularizer = L2SideBoxRegularizer(log_scale=True, weight=1.0)

        # Initialize LPIPS loss
        if "lpips_loss" in self.loss_weights and self.loss_weights["lpips_loss"] > 0:
            self.lpips_loss_fn = lpips.LPIPS(net="vgg")
        else:
            self.lpips_loss_fn = None

        # Initialize SSIM loss
        # Using win_size=11, data_range=1 (for normalized [0,1] range), channel=3 (RGB)
        if "ssim_loss" in self.loss_weights and self.loss_weights["ssim_loss"] > 0:
            self.ssim_loss_fn = SSIM(
                win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
            )
        else:
            self.ssim_loss_fn = None

        self.viz_datapoint = None

    def forward(self, x):
        images = x["images"]
        object_masks = x["object_masks"]
        batch_size = images.shape[0]

        image_patches = self.divide_image_into_patches(
            images
        )  # (batch_size, grid_h, grid_w, C, patch_h, patch_w)
        mask_patches = self.divide_image_into_patches(
            object_masks.unsqueeze(1)
        )  # (batch_size, grid_h, grid_w, 1, patch_h, patch_w)
        # batch_bounding_boxes = self.compute_bounding_boxes(batch_mask_patches) # (batch_size, 4)
        _, grid_h, grid_w, C, patch_h, patch_w = image_patches.shape
        num_patches = grid_h * grid_w

        # Compute valid patch mask if crop_objects is enabled
        if self.crop_objects:
            valid_patch_mask = self.compute_valid_patch_mask(
                mask_patches
            )  # (batch_size, grid_h, grid_w)
            # Stitch valid patches to create cropped full image
            cropped_full_image = self.stitch_patches(
                image_patches, valid_patch_mask
            )  # (batch_size, C, image_size, image_size)
        else:
            valid_patch_mask = None
            cropped_full_image = None

        image_patches_flattened = image_patches.reshape(
            batch_size * num_patches, C, patch_h, patch_w
        )  # (batch_size * num_patches, C, patch_h, patch_w)
        image_patches_resized = F.interpolate(
            image_patches_flattened,
            size=(self.image_size[0], self.image_size[1]),
            mode="bilinear",
        )  # (batch_size * num_patches, C, self.image_size[0], self.image_size[1])
        image_patches_reshaped = image_patches_resized.reshape(
            batch_size,
            num_patches,
            C,
            self.image_size[0],
            self.image_size[1],
        )  # (batch_size, num_patches, C, self.image_size[0], self.image_size[1])

        # Use cropped full image if crop_objects is enabled, otherwise use original image
        if self.crop_objects:
            full_image_for_encoder = cropped_full_image  # (batch_size, C, image_size, image_size)
        else:
            full_image_for_encoder = images  # (batch_size, C, H, W)

        all_images = torch.cat(
            [image_patches_reshaped, full_image_for_encoder.unsqueeze(1)],
            dim=1,
        )  # (batch_size, num_patches + 1, C, self.image_size[0], self.image_size[1])
        all_images_flattened = all_images.reshape(
            -1, C, self.image_size[0], self.image_size[1]
        )  # ((batch_size * (num_patches + 1)), C, self.image_size[0], self.image_size[1])

        all_mu_min, all_mu_max = self.vae.encode(
            all_images_flattened
        )  # (batch_size * (num_patches + 1), self.embed_dim) , (batch_size * (num_patches + 1), self.embed_dim)
        all_z = torch.cat(
            [all_mu_min, all_mu_max], dim=-1
        )  # (batch_size * (num_patches + 1), self.box_embed_dim)
        all_reconstructed_images = self.vae.decode(
            all_z
        )  # ((batch_size * (num_patches + 1)), C, self.image_size[0], self.image_size[1])

        all_mu_min = all_mu_min.reshape(
            batch_size, num_patches + 1, self.embed_dim
        )  # (batch_size, num_patches + 1, self.embed_dim)
        all_mu_max = all_mu_max.reshape(
            batch_size, num_patches + 1, self.embed_dim
        )  # (batch_size, num_patches + 1, self.embed_dim)
        all_z = all_z.reshape(
            batch_size, num_patches + 1, self.box_embed_dim
        )  # (batch_size, num_patches + 1, self.box_embed_dim)
        all_reconstructed_images = all_reconstructed_images.reshape(
            batch_size,
            num_patches + 1,
            C,
            self.image_size[0],
            self.image_size[1],
        )  # (batch_size, num_patches + 1, C, self.image_size[0], self.image_size[1])

        return {
            "mu_min": all_mu_min,
            "mu_max": all_mu_max,
            "z": all_z,
            "images": all_images,
            "reconstructed_images": all_reconstructed_images,
            "valid_patch_mask": valid_patch_mask,  # (batch_size, grid_h, grid_w) or None
        }

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        num_patches = self.grid_size[0] * self.grid_size[1]
        batch_size = outputs["reconstructed_images"].shape[0]

        reconstructed_patches = outputs["reconstructed_images"][:, :-1, :, :, :].reshape(
            -1, 3, self.image_size[0], self.image_size[1]
        )  # (batch_size * num_patches, 3, self.image_size[0], self.image_size[1])
        reconstructed_full_image = outputs["reconstructed_images"][:, -1, :, :, :].reshape(
            -1, 3, self.image_size[0], self.image_size[1]
        )  # (batch_size, 3, self.image_size[0], self.image_size[1])
        original_patches = outputs["images"][:, :-1, :, :, :].reshape(
            -1, 3, self.image_size[0], self.image_size[1]
        )  # (batch_size * num_patches, 3, self.image_size[0], self.image_size[1])
        original_full_image = outputs["images"][:, -1, :, :, :].reshape(
            -1, 3, self.image_size[0], self.image_size[1]
        )  # (batch_size, 3, self.image_size[0], self.image_size[1])

        # Create mask for valid patches if crop_objects is enabled
        if self.crop_objects and outputs["valid_patch_mask"] is not None:
            valid_mask = outputs["valid_patch_mask"].reshape(-1)  # (batch_size * num_patches)
            valid_indices = valid_mask > 0
            num_valid_patches = valid_indices.sum().item()
        else:
            valid_mask = None
            valid_indices = None
            num_valid_patches = batch_size * num_patches

        if self.viz_datapoint is None:
            self.viz_datapoint = {
                key: value[0].cpu() for key, value in outputs.items() if key != "valid_patch_mask"
            }

        all_box_embeddings = torch.stack(
            [outputs["mu_min"], outputs["mu_max"]], dim=2
        )  # (batch_size, num_patches + 1, 2, self.embed_dim)
        all_box_tensors = BoxTensor(all_box_embeddings)
        patch_box_tensors = BoxTensor(all_box_embeddings[:, :-1, :, :])
        image_box_tensor = BoxTensor(all_box_embeddings[:, -1, :, :])

        # Compute patch intersection, considering only valid patches if crop_objects is enabled
        if self.crop_objects and outputs["valid_patch_mask"] is not None:
            # Compute intersection only for valid patches
            # Collect box embeddings (min, max) for each batch element
            intersection_embeddings_list = []
            for b in range(batch_size):
                valid_mask_b = outputs["valid_patch_mask"][b].reshape(-1)  # (num_patches,)
                valid_patch_indices = torch.nonzero(valid_mask_b, as_tuple=True)[0]

                if len(valid_patch_indices) > 0:
                    # Start with first valid patch
                    intersection_b = patch_box_tensors[b : b + 1, valid_patch_indices[0]]
                    # Intersect with other valid patches
                    for idx in valid_patch_indices[1:]:
                        intersection_b = self.box_intersection(
                            intersection_b,
                            patch_box_tensors[b : b + 1, idx],
                        )
                    # Extract the box embedding (shape: [1, 2, embed_dim])
                    intersection_embeddings_list.append(
                        torch.stack([intersection_b.z, intersection_b.Z], dim=1)
                    )
                else:
                    # No valid patches, use first patch as placeholder
                    placeholder_box = patch_box_tensors[b : b + 1, 0]
                    intersection_embeddings_list.append(
                        torch.stack([placeholder_box.z, placeholder_box.Z], dim=1)
                    )

            # Stack all embeddings and create BoxTensor (batch_size, 2, embed_dim)
            intersection_embeddings = torch.cat(intersection_embeddings_list, dim=0)
            patch_intersection_box = BoxTensor(intersection_embeddings)
        else:
            # Original behavior: intersect all patches
            patch_intersection_box = patch_box_tensors[:, 0]
            for idx in range(1, num_patches):
                patch_intersection_box = self.box_intersection(
                    patch_intersection_box,
                    patch_box_tensors[:, idx],
                )

        image_box_volume = self.box_volume(image_box_tensor)
        image_patch_intersection_volume = self.box_volume(
            self.box_intersection(image_box_tensor, patch_intersection_box)
        )
        patch_intersection_volume = self.box_volume(patch_intersection_box)

        # reconstruction_loss = (
        #     F.mse_loss(
        #         outputs["reconstructed_images"],
        #         outputs["images"],
        #     )
        #     * self.loss_weights["reconstruction_loss"]
        # )

        # Compute patch reconstruction loss only for valid patches if crop_objects is enabled
        if self.crop_objects and valid_indices is not None and num_valid_patches > 0:
            valid_reconstructed_patches = reconstructed_patches[valid_indices]
            valid_original_patches = original_patches[valid_indices]
            patch_reconstruction_loss = F.mse_loss(
                valid_reconstructed_patches, valid_original_patches
            )
        else:
            patch_reconstruction_loss = F.mse_loss(reconstructed_patches, original_patches)

        full_image_reconstruction_loss = F.mse_loss(reconstructed_full_image, original_full_image)
        if "full_image_weight" in self.loss_weights and self.loss_weights["full_image_weight"] > 0:
            reconstruction_loss = (
                patch_reconstruction_loss
                + (full_image_reconstruction_loss * self.loss_weights["full_image_weight"])
            ) * self.loss_weights["reconstruction_loss"]
        else:
            reconstruction_loss = (
                patch_reconstruction_loss + full_image_reconstruction_loss
            ) * self.loss_weights["reconstruction_loss"]

        inclusion_loss = (
            image_box_volume - image_patch_intersection_volume
        ).mean() * self.loss_weights["inclusion_loss"]
        if (
            "patch_tightness_loss" in self.loss_weights
            and self.loss_weights["patch_tightness_loss"] > 0
        ):
            patch_tightness_loss = (
                patch_intersection_volume - image_patch_intersection_volume
            ).mean() * self.loss_weights["patch_tightness_loss"]
        else:
            patch_tightness_loss = torch.tensor(0.0, device=self.device)
        box_volume_regularization_loss = (
            self.box_volume_regularizer(all_box_tensors).mean()
            * self.loss_weights["box_volume_regularization_loss"]
        )
        min_side_regularization_loss = (
            self.min_side_regularization(all_box_tensors)
            * self.loss_weights["min_side_regularization_loss"]
        )

        # Calculate LPIPS loss if weight is provided
        if "lpips_loss" in self.loss_weights and self.loss_weights["lpips_loss"] > 0:
            # Compute LPIPS loss only for valid patches if crop_objects is enabled
            if self.crop_objects and valid_indices is not None and num_valid_patches > 0:
                # For patches, only use valid ones
                valid_reconstructed_patches = reconstructed_patches[valid_indices]
                valid_original_patches = original_patches[valid_indices]

                # Scale to [-1, 1] for LPIPS
                valid_reconstructed_patches = (valid_reconstructed_patches * 2.0) - 1.0
                valid_original_patches = (valid_original_patches * 2.0) - 1.0
                reconstructed_full_image_scaled = (reconstructed_full_image * 2.0) - 1.0
                original_full_image_scaled = (original_full_image * 2.0) - 1.0

                # Compute LPIPS for valid patches and full image
                lpips_patches = self.lpips_loss_fn(
                    valid_reconstructed_patches, valid_original_patches
                ).mean()
                lpips_full = self.lpips_loss_fn(
                    reconstructed_full_image_scaled, original_full_image_scaled
                ).mean()

                # Combine similar to reconstruction loss
                if (
                    "full_image_weight" in self.loss_weights
                    and self.loss_weights["full_image_weight"] > 0
                ):
                    lpips_loss = (
                        lpips_patches + lpips_full * self.loss_weights["full_image_weight"]
                    ) * self.loss_weights["lpips_loss"]
                else:
                    lpips_loss = (lpips_patches + lpips_full) * self.loss_weights["lpips_loss"]
            else:
                reconstructed_flat = outputs["reconstructed_images"].reshape(
                    -1, 3, self.image_size[0], self.image_size[1]
                )
                images_flat = outputs["images"].reshape(
                    -1, 3, self.image_size[0], self.image_size[1]
                )

                reconstructed_flat = (reconstructed_flat * 2.0) - 1.0
                images_flat = (images_flat * 2.0) - 1.0

                lpips_loss = (
                    self.lpips_loss_fn(reconstructed_flat, images_flat).mean()
                    * self.loss_weights["lpips_loss"]
                )
        else:
            lpips_loss = torch.tensor(0.0, device=self.device)

        # Calculate SSIM loss if weight is provided
        if "ssim_loss" in self.loss_weights and self.loss_weights["ssim_loss"] > 0:

            # Compute SSIM loss only for valid patches if crop_objects is enabled
            if self.crop_objects and valid_indices is not None and num_valid_patches > 0:
                valid_reconstructed_patches = reconstructed_patches[valid_indices]
                valid_original_patches = original_patches[valid_indices]
                ssim_loss_patches = 1.0 - self.ssim_loss_fn(
                    valid_reconstructed_patches, valid_original_patches
                )
            else:
                ssim_loss_patches = 1.0 - self.ssim_loss_fn(reconstructed_patches, original_patches)

            ssim_loss_full_image = 1.0 - self.ssim_loss_fn(
                reconstructed_full_image, original_full_image
            )

            if (
                "full_image_weight" in self.loss_weights
                and self.loss_weights["full_image_weight"] > 0
            ):
                ssim_loss = (
                    ssim_loss_patches
                    + (ssim_loss_full_image * self.loss_weights["full_image_weight"])
                ) * self.loss_weights["ssim_loss"]

            else:
                ssim_loss = (ssim_loss_patches + ssim_loss_full_image) * self.loss_weights[
                    "ssim_loss"
                ]

            # reconstructed_flat = outputs["reconstructed_images"].reshape(
            #     -1, 3, self.image_size[0], self.image_size[1]
            # )
            # images_flat = outputs["images"].reshape(-1, 3, self.image_size[0], self.image_size[1])

            # ssim_value = self.ssim_loss_fn(reconstructed_flat, images_flat)
            # ssim_loss = (1.0 - ssim_value) * self.loss_weights["ssim_loss"]
        else:
            ssim_loss = torch.tensor(0.0, device=self.device)

        # Calculate supervised contrastive loss (intersection) if weight is provided
        if (
            "supervised_contrastive_loss_intersection" in self.loss_weights
            and self.loss_weights["supervised_contrastive_loss_intersection"] > 0
        ):
            supervised_contrastive_loss_intersection = (
                self.compute_supervised_contrastive_loss_intersection(image_box_tensor, batch)
            )
        else:
            supervised_contrastive_loss_intersection = torch.tensor(0.0, device=self.device)

        # Calculate supervised contrastive loss (distance) if weight is provided
        if (
            "supervised_contrastive_loss_distance" in self.loss_weights
            and self.loss_weights["supervised_contrastive_loss_distance"] > 0
        ):
            supervised_contrastive_loss_distance = (
                self.compute_supervised_contrastive_loss_distance(image_box_tensor, batch)
            )
        else:
            supervised_contrastive_loss_distance = torch.tensor(0.0, device=self.device)

        total_loss = (
            reconstruction_loss
            + inclusion_loss
            + patch_tightness_loss
            + box_volume_regularization_loss
            + min_side_regularization_loss
            + lpips_loss
            + ssim_loss
            + supervised_contrastive_loss_intersection
            + supervised_contrastive_loss_distance
        )

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "patch_reconstruction_loss": patch_reconstruction_loss,
            "full_image_reconstruction_loss": full_image_reconstruction_loss,
            "inclusion_loss": inclusion_loss,
            "patch_tightness_loss": patch_tightness_loss,
            "box_volume_regularization_loss": box_volume_regularization_loss,
            "min_side_regularization_loss": min_side_regularization_loss,
            "total_loss": total_loss,
        }

        if "lpips_loss" in self.loss_weights and self.loss_weights["lpips_loss"] > 0:
            loss_dict["lpips_loss"] = lpips_loss

        if "ssim_loss" in self.loss_weights and self.loss_weights["ssim_loss"] > 0:
            loss_dict["ssim_loss"] = ssim_loss
            loss_dict["ssim_loss_patches"] = ssim_loss_patches
            loss_dict["ssim_loss_full_image"] = ssim_loss_full_image

        if (
            "supervised_contrastive_loss_intersection" in self.loss_weights
            and self.loss_weights["supervised_contrastive_loss_intersection"] > 0
        ):
            loss_dict["supervised_contrastive_loss_intersection"] = (
                supervised_contrastive_loss_intersection
            )

        if (
            "supervised_contrastive_loss_distance" in self.loss_weights
            and self.loss_weights["supervised_contrastive_loss_distance"] > 0
        ):
            loss_dict["supervised_contrastive_loss_distance"] = supervised_contrastive_loss_distance

        # Log number of valid patches if crop_objects is enabled
        if self.crop_objects:
            loss_dict["num_valid_patches"] = float(num_valid_patches)
            loss_dict["valid_patch_ratio"] = float(num_valid_patches) / float(
                batch_size * num_patches
            )

        self.log_dict(loss_dict, prog_bar=True, on_epoch=True)

        return total_loss

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
            grid_h, grid_w = self.grid_size
            fig = create_reconstruction_visualization(
                self.viz_datapoint, (grid_h, grid_w), self.image_size, self.current_epoch + 1
            )
            self.logger.experiment.log({"reconstruction_visualization": fig})
            self.viz_datapoint = None
            plt.close(fig)

    def divide_image_into_patches(self, images):
        B, C, H, W = images.shape

        grid_h, grid_w = self.grid_size
        patch_h = H // grid_h
        patch_w = W // grid_w

        if H % grid_h != 0 or W % grid_w != 0:
            raise ValueError("Image dimensions must be divisible by grid size.")

        patches = images.view(
            B, C, grid_h, patch_h, grid_w, patch_w
        )  # (B, C, grid_h, patch_h, grid_w, patch_w)
        patches = patches.permute(0, 2, 4, 1, 3, 5)  # (B, grid_h, grid_w, C, patch_h, patch_w)
        patches = patches.contiguous()

        return patches

    def compute_valid_patch_mask(self, mask_patches):
        """
        Compute a binary mask indicating which patches contain the object.

        Args:
            mask_patches: (batch_size, grid_h, grid_w, 1, patch_h, patch_w)

        Returns:
            valid_mask: (batch_size, grid_h, grid_w) binary mask where 1 indicates valid patch
        """
        batch_size, grid_h, grid_w, _, patch_h, patch_w = mask_patches.shape
        # Sum over spatial dimensions of each patch
        patch_sums = mask_patches.sum(dim=(-3, -2, -1))  # (batch_size, grid_h, grid_w)
        valid_mask = (patch_sums > 0).float()  # (batch_size, grid_h, grid_w)
        return valid_mask

    def stitch_patches(self, image_patches, valid_mask):
        """
        Stitch valid patches together to create a cropped full image.

        Args:
            image_patches: (batch_size, grid_h, grid_w, C, patch_h, patch_w)
            valid_mask: (batch_size, grid_h, grid_w) binary mask

        Returns:
            stitched_image: (batch_size, C, stitched_h, stitched_w)
        """
        batch_size, grid_h, grid_w, C, patch_h, patch_w = image_patches.shape

        # Find bounding box of valid patches for each sample in the batch
        stitched_images = []
        for b in range(batch_size):
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=False)  # (num_valid, 2)

            if valid_indices.shape[0] == 0:
                # No valid patches, return a single patch of zeros
                stitched_images.append(
                    torch.zeros(C, patch_h, patch_w, device=image_patches.device)
                )
                continue

            # Find bounding box of valid patches
            min_row = valid_indices[:, 0].min().item()
            max_row = valid_indices[:, 0].max().item()
            min_col = valid_indices[:, 1].min().item()
            max_col = valid_indices[:, 1].max().item()

            # Extract the bounding box region
            bbox_patches = image_patches[
                b, min_row : max_row + 1, min_col : max_col + 1
            ]  # (bbox_h, bbox_w, C, patch_h, patch_w)
            bbox_h, bbox_w = bbox_patches.shape[0], bbox_patches.shape[1]

            # Reshape to stitch patches together
            bbox_patches = bbox_patches.permute(
                2, 0, 3, 1, 4
            )  # (C, bbox_h, patch_h, bbox_w, patch_w)
            stitched = bbox_patches.reshape(
                C, bbox_h * patch_h, bbox_w * patch_w
            )  # (C, stitched_h, stitched_w)
            stitched_images.append(stitched)

        # Resize all stitched images to the target image size
        stitched_batch = []
        for stitched in stitched_images:
            resized = F.interpolate(
                stitched.unsqueeze(0),
                size=(self.image_size[0], self.image_size[1]),
                mode="bilinear",
            ).squeeze(
                0
            )  # (C, self.image_size[0], self.image_size[1])
            stitched_batch.append(resized)

        stitched_batch = torch.stack(
            stitched_batch, dim=0
        )  # (batch_size, C, self.image_size[0], self.image_size[1])
        return stitched_batch

    def min_side_regularization(self, box_tensors):
        side_lengths = box_tensors.Z - box_tensors.z
        penalty = F.relu(self.min_side_length - side_lengths)
        penalty = penalty.sum(dim=-1).mean()
        return penalty

    def compute_supervised_contrastive_loss_intersection(self, image_box_tensor, batch):
        """
        Compute supervised contrastive loss that penalizes intersection volume
        between boxes of different shapes (circle vs square).

        Args:
            image_box_tensor: BoxTensor of shape (batch_size, 2, embed_dim) representing
                            the full image box embeddings
            batch: Batch dictionary that should contain "metadata" with shape information

        Returns:
            loss: Scalar tensor representing the supervised contrastive loss
        """
        if "metadata" not in batch:
            return torch.tensor(0.0, device=self.device)

        batch_size = image_box_tensor.z.shape[0]
        metadata = batch["metadata"]

        # Extract shapes from metadata
        # metadata is a dict with 'shape' key containing a list of shape strings
        shapes = metadata["shape"]

        # Find indices for circles and squares
        circle_indices = [i for i, shape in enumerate(shapes) if shape == "circle"]
        square_indices = [i for i, shape in enumerate(shapes) if shape == "square"]

        if len(circle_indices) == 0 or len(square_indices) == 0:
            # Need both shapes in the batch to compute contrastive loss
            return torch.tensor(0.0, device=self.device)

        # Extract circle and square box tensors
        # Stack min and max to create (num_circles, 2, embed_dim) format
        circle_embeddings = torch.stack(
            [image_box_tensor.z[circle_indices], image_box_tensor.Z[circle_indices]], dim=1
        )  # (num_circles, 2, embed_dim)
        square_embeddings = torch.stack(
            [image_box_tensor.z[square_indices], image_box_tensor.Z[square_indices]], dim=1
        )  # (num_squares, 2, embed_dim)

        circle_boxes = BoxTensor(circle_embeddings)
        square_boxes = BoxTensor(square_embeddings)

        # Compute intersection volumes between all circle-square pairs
        intersection_volumes = []
        for i in range(len(circle_indices)):
            for j in range(len(square_indices)):
                # Compute intersection between circle i and square j
                intersection_box = self.box_intersection(
                    circle_boxes[i : i + 1], square_boxes[j : j + 1]
                )
                intersection_volume = self.box_volume(intersection_box)
                # Clamp to non-negative: volume should never be negative, but numerical issues
                # with log_scale or invalid intersections might cause negative values
                intersection_volume = torch.clamp(intersection_volume, min=0.0)
                intersection_volumes.append(intersection_volume)

        if len(intersection_volumes) > 0:
            # Sum all intersection volumes and penalize them
            total_intersection_volume = torch.stack(intersection_volumes).sum()
            supervised_contrastive_loss = (
                total_intersection_volume
                * self.loss_weights["supervised_contrastive_loss_intersection"]
            )
        else:
            supervised_contrastive_loss = torch.tensor(0.0, device=self.device)

        return supervised_contrastive_loss

    # def compute_supervised_contrastive_loss_distance(self, image_box_tensor, batch):
    #     """
    #     Compute supervised contrastive loss that pushes centroids of boxes
    #     with different shapes (circle vs square) away from each other by
    #     minimizing the negative squared distance between centroids.

    #     If a margin is specified in loss_weights["supervised_contrastive_loss_distance_margin"],
    #     pairs with distance >= margin have zero loss (already far enough apart).

    #     Args:
    #         image_box_tensor: BoxTensor of shape (batch_size, 2, embed_dim) representing
    #                         the full image box embeddings
    #         batch: Batch dictionary that should contain "metadata" with shape information

    #     Returns:
    #         loss: Scalar tensor representing the supervised contrastive loss (negative distance,
    #               zero for pairs beyond the margin if margin is specified)
    #     """
    #     if "metadata" not in batch:
    #         return torch.tensor(0.0, device=self.device)

    #     batch_size = image_box_tensor.z.shape[0]
    #     metadata = batch["metadata"]

    #     # Extract shapes from metadata
    #     # metadata is a dict with 'shape' key containing a list of shape strings
    #     shapes = metadata["shape"]

    #     # Find indices for circles and squares
    #     circle_indices = [i for i, shape in enumerate(shapes) if shape == "circle"]
    #     square_indices = [i for i, shape in enumerate(shapes) if shape == "square"]

    #     if len(circle_indices) == 0 or len(square_indices) == 0:
    #         # Need both shapes in the batch to compute contrastive loss
    #         return torch.tensor(0.0, device=self.device)

    #     # Compute centroids: (z + Z) / 2
    #     circle_centroids = (
    #         image_box_tensor.z[circle_indices] + image_box_tensor.Z[circle_indices]
    #     ) / 2.0  # (num_circles, embed_dim)
    #     square_centroids = (
    #         image_box_tensor.z[square_indices] + image_box_tensor.Z[square_indices]
    #     ) / 2.0  # (num_squares, embed_dim)

    #     # Get margin value if provided, otherwise use infinity (no margin)
    #     margin = self.loss_weights.get("supervised_contrastive_loss_distance_margin", float("inf"))
    #     margin_squared = margin**2  # Compare squared distances to squared margin for efficiency

    #     # Compute squared Euclidean distances between all circle-square centroid pairs
    #     distances_squared = []
    #     for i in range(len(circle_indices)):
    #         for j in range(len(square_indices)):
    #             # Compute squared Euclidean distance between centroids
    #             diff = circle_centroids[i] - square_centroids[j]  # (embed_dim,)
    #             dist_sq = torch.sum(diff**2)  # Scalar
    #             distances_squared.append(dist_sq)

    #     if len(distances_squared) > 0:
    #         all_distances_sq = torch.stack(distances_squared)
    #         # Apply margin: only penalize distances below the margin
    #         # For distances >= margin, the loss contribution should be 0
    #         # Use negative squared distance to maximize separation (minimizing this maximizes distance)
    #         loss_per_pair = -all_distances_sq  # Negative to push apart
    #         negative_margin_squared = -margin_squared
    #         # Zero out loss for pairs where distance >= margin (already far enough apart)
    #         # Compare loss (which is negative) to negative margin:
    #         # If distance < margin: distance^2 < margin^2, so -distance^2 > -margin^2
    #         #   This means loss_per_pair > negative_margin_squared, so we keep the loss
    #         # If distance >= margin: distance^2 >= margin^2, so -distance^2 <= -margin^2
    #         #   This means loss_per_pair <= negative_margin_squared, so we set to 0
    #         loss_per_pair = torch.where(
    #             loss_per_pair > negative_margin_squared,  # Keep loss when distance < margin
    #             loss_per_pair,  # Keep negative distance if below margin (push apart)
    #             torch.zeros_like(loss_per_pair),  # Zero loss if above margin (already far enough)
    #         )
    #         supervised_contrastive_loss = (
    #             loss_per_pair.mean() * self.loss_weights["supervised_contrastive_loss_distance"]
    #         )
    #     else:
    #         supervised_contrastive_loss = torch.tensor(0.0, device=self.device)

    #     return supervised_contrastive_loss

    def compute_supervised_contrastive_loss_distance(self, image_box_tensor, batch):
        if "metadata" not in batch:
            return torch.tensor(0.0, device=self.device)

        metadata = batch["metadata"]
        shapes = metadata["shape"]

        # 1. Identification (Same as before)
        circle_indices = [i for i, shape in enumerate(shapes) if shape == "circle"]
        square_indices = [i for i, shape in enumerate(shapes) if shape == "square"]

        if len(circle_indices) == 0 or len(square_indices) == 0:
            return torch.tensor(0.0, device=self.device)

        # 2. Compute Centroids (Same as before)
        # Shape: (num_circles, embed_dim)
        circle_centroids = (
            image_box_tensor.z[circle_indices] + image_box_tensor.Z[circle_indices]
        ) / 2.0
        # Shape: (num_squares, embed_dim)
        square_centroids = (
            image_box_tensor.z[square_indices] + image_box_tensor.Z[square_indices]
        ) / 2.0

        # 3. Vectorized Distance Calculation (Replaces nested loops)
        # We want pairwise distances between all circles and all squares.
        # Broadcasting: (N, 1, D) - (1, M, D) -> (N, M, D)
        diff = circle_centroids.unsqueeze(1) - square_centroids.unsqueeze(0)

        # Squared Euclidean distance
        # Shape: (num_circles, num_squares)
        dist_sq = torch.sum(diff**2, dim=-1)

        # 4. Margin Logic (Corrected to Hinge Loss)
        margin = self.loss_weights.get("supervised_contrastive_loss_distance_margin", 1.0)

        # Logic: We want dist_sq >= margin^2.
        # Loss = ReLU(margin^2 - dist_sq)
        # If dist_sq is small (0), Loss is high (margin^2).
        # If dist_sq > margin^2, Loss is 0.
        margin_sq = margin**2

        # This creates a smooth gradient pushing them apart until they hit the margin
        hinge_loss = F.relu(margin_sq - dist_sq)

        # 5. Aggregate and Weight
        loss = hinge_loss.mean() * self.loss_weights["supervised_contrastive_loss_distance"]

        return loss
