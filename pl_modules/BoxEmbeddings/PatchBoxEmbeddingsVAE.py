import torch
from torch import nn
import lightning as L
import torch.nn.functional as F
import numpy as np
import wandb
import lpips

from box_embeddings.parameterizations.box_tensor import (
    BoxTensor,
)
from box_embeddings.modules.volume import Volume
from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.regularization import (
    L2SideBoxRegularizer,
)

from .utils import create_reconstruction_visualization


class VanillaVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dims):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        encoder_modules = []
        in_channels = 3
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)
        self.fc_mu_min = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_mu_max = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim * 2, hidden_dims[-1] * 4)

        decoder_modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1],
                out_channels=3,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu_min = self.fc_mu_min(result)
        mu_max = self.fc_mu_max(result)

        return [mu_min, mu_max]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input, **kwargs):
        mu_min, mu_max = self.encode(input)
        z = torch.cat([mu_min, mu_max], dim=-1)  # (batch_size, self.box_embed_dim)
        return [self.decode(z), input, mu_min, mu_max]


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

        self.loss_weights = config["model"]["config"]["loss_weights"]

        self.vae = VanillaVAE(self.embed_dim, self.hidden_dims)
        self.box_volume = Volume(
            volume_temperature=self.gumbel_temp,
            log_scale=True,
        )
        self.box_intersection = Intersection(intersection_temperature=self.gumbel_temp)
        self.box_volume_regularizer = L2SideBoxRegularizer(log_scale=True, weight=1.0)

        # Initialize LPIPS loss
        self.lpips_loss_fn = lpips.LPIPS(net="vgg")

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
        all_images = torch.cat(
            [image_patches_reshaped, images.unsqueeze(1)],
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
        }

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        num_patches = self.grid_size[0] * self.grid_size[1]

        if self.viz_datapoint is None:
            self.viz_datapoint = {key: value[0].cpu() for key, value in outputs.items()}

        all_box_embeddings = torch.stack(
            [outputs["mu_min"], outputs["mu_max"]], dim=2
        )  # (batch_size, num_patches + 1, 2, self.embed_dim)
        all_box_tensors = BoxTensor(all_box_embeddings)
        patch_box_tensors = BoxTensor(all_box_embeddings[:, :-1, :, :])
        image_box_tensor = BoxTensor(all_box_embeddings[:, -1, :, :])

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

        reconstruction_loss = (
            F.mse_loss(
                outputs["reconstructed_images"],
                outputs["images"],
            )
            * self.loss_weights["reconstruction_loss"]
        )
        inclusion_loss = (
            image_box_volume - image_patch_intersection_volume
        ).mean() * self.loss_weights["inclusion_loss"]
        box_volume_regularization_loss = (
            self.box_volume_regularizer(all_box_tensors).mean()
            * self.loss_weights["box_volume_regularization_loss"]
        )
        min_side_regularization_loss = (
            self.min_side_regularization(all_box_tensors)
            * self.loss_weights["min_side_regularization_loss"]
        )

        # Calculate LPIPS loss if weight is provided
        if "lpips_loss" in self.loss_weights:
            # Flatten the batch dimension with num_patches+1 dimension for LPIPS
            batch_size = outputs["reconstructed_images"].shape[0]
            num_images = outputs["reconstructed_images"].shape[1]
            reconstructed_flat = outputs["reconstructed_images"].reshape(
                -1, 3, self.image_size[0], self.image_size[1]
            )
            images_flat = outputs["images"].reshape(-1, 3, self.image_size[0], self.image_size[1])

            # LPIPS expects inputs in range [-1, 1], which our Tanh output already provides
            lpips_loss_value = self.lpips_loss_fn(reconstructed_flat, images_flat).mean()
            lpips_loss = lpips_loss_value * self.loss_weights["lpips_loss"]
        else:
            lpips_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            reconstruction_loss
            + inclusion_loss
            + box_volume_regularization_loss
            + min_side_regularization_loss
            + lpips_loss
        )

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "inclusion_loss": inclusion_loss,
            "box_volume_regularization_loss": box_volume_regularization_loss,
            "min_side_regularization_loss": min_side_regularization_loss,
            "total_loss": total_loss,
        }

        if "lpips_loss" in self.loss_weights:
            loss_dict["lpips_loss"] = lpips_loss
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

    def min_side_regularization(self, box_tensors):
        side_lengths = box_tensors.Z - box_tensors.z
        penalty = F.relu(self.min_side_length - side_lengths)
        penalty = penalty.sum(dim=-1).mean()
        return penalty
