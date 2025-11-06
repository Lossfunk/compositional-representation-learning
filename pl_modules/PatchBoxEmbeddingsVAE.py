import torch
from torch import nn
import lightning as L
import torch.nn.functional as F

from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.volume import Volume
from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.regularization import L2SideBoxRegularizer

class VanillaVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dims):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        encoder_modules = []
        # Build Encoder
        in_channels = 3
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)
        self.fc_mu_min = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_mu_max = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        decoder_modules = []

        self.decoder_input = nn.Linear(latent_dim * 2, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu_min = self.fc_mu_min(result)
        mu_max = self.fc_mu_max(result)

        return [mu_min, mu_max]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    # def reparameterize(self, mu_min, mu_max):
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, input, **kwargs):
        mu_min, mu_max = self.encode(input)
        # z = self.reparameterize(mu, log_var)
        z = torch.cat([mu_min, mu_max], dim=-1) # (batch_size, self.box_embed_dim)
        return  [self.decode(z), input, mu_min, mu_max]

    # def loss_function(self,
    #                   *args,
    #                   **kwargs):
    #     """
    #     Computes the VAE loss function.
    #     KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    #     :param args:
    #     :param kwargs:
    #     :return:
    #     """
    #     recons = args[0]
    #     input = args[1]
    #     mu = args[2]
    #     log_var = args[3]

    #     kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
    #     recons_loss =F.mse_loss(recons, input)


    #     kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    #     loss = recons_loss + kld_weight * kld_loss
    #     return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs):
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples

    # def generate(self, x, **kwargs):
    #     """
    #     Given an input image x, returns the reconstructed image
    #     :param x: (Tensor) [B x C x H x W]
    #     :return: (Tensor) [B x C x H x W]
    #     """

    #     return self.forward(x)[0]

class PatchBoxEmbeddingsVAE(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config['model']['config']['embed_dim']
        self.box_embed_dim = self.embed_dim * 2
        self.hidden_dims = config['model']['config']['hidden_dims']
        self.grid_size = config['model']['config']['grid_size']
        self.image_size = config['data']['train']['config']['image_size']
        self.gumbel_temp = config['model']['config']['gumbel_temp']

        self.loss_weights = config['model']['config']['loss_weights']

        self.vae = VanillaVAE(self.embed_dim, self.hidden_dims)
        self.box_volume = Volume(volume_temperature=self.gumbel_temp, log_scale=True)
        self.box_intersection = Intersection(intersection_temperature=self.gumbel_temp)
        self.box_volume_regularizer = L2SideBoxRegularizer(log_scale=True, weight=1.0)

    def forward(self, x):
        images = x['images']
        object_masks = x['object_masks']
        batch_size = images.shape[0]

        image_patches = self.divide_image_into_patches(images) # (batch_size, grid_h, grid_w, C, patch_h, patch_w)
        mask_patches = self.divide_image_into_patches(object_masks.unsqueeze(1)) # (batch_size, grid_h, grid_w, 1, patch_h, patch_w)
        # batch_bounding_boxes = self.compute_bounding_boxes(batch_mask_patches) # (batch_size, 4)
        _, grid_h, grid_w, C, patch_h, patch_w = image_patches.shape
        num_patches = grid_h * grid_w
        
        image_patches_flattened = image_patches.reshape(batch_size * num_patches, C, patch_h, patch_w) # (batch_size * num_patches, C, patch_h, patch_w)
        image_patches_resized = F.interpolate(image_patches_flattened, size=(self.image_size[0], self.image_size[1]), mode='bilinear') # (batch_size * num_patches, C, self.image_size[0], self.image_size[1])
        image_patches_reshaped = image_patches_resized.reshape(batch_size, num_patches, C, self.image_size[0], self.image_size[1]) # (batch_size, num_patches, C, self.image_size[0], self.image_size[1])
        all_images = torch.cat([image_patches_reshaped, images.unsqueeze(1)], dim=1) # (batch_size, num_patches + 1, C, self.image_size[0], self.image_size[1])
        all_images_flattened = all_images.reshape(-1, C, self.image_size[0], self.image_size[1]) # ((batch_size * (num_patches + 1)), C, self.image_size[0], self.image_size[1])

        all_mu_min, all_mu_max = self.vae.encode(all_images_flattened) # (batch_size * (num_patches + 1), self.embed_dim) , (batch_size * (num_patches + 1), self.embed_dim)
        all_z = torch.cat([all_mu_min, all_mu_max], dim=-1) # (batch_size * (num_patches + 1), self.box_embed_dim)
        all_reconstructed_images = self.vae.decode(all_z) # ((batch_size * (num_patches + 1)), C, self.image_size[0], self.image_size[1])

        all_mu_min = all_mu_min.reshape(batch_size, num_patches + 1, self.embed_dim) # (batch_size, num_patches + 1, self.embed_dim)
        all_mu_max = all_mu_max.reshape(batch_size, num_patches + 1, self.embed_dim) # (batch_size, num_patches + 1, self.embed_dim)
        all_z = all_z.reshape(batch_size, num_patches + 1, self.box_embed_dim) # (batch_size, num_patches + 1, self.box_embed_dim)
        all_reconstructed_images = all_reconstructed_images.reshape(batch_size, num_patches + 1, C, self.image_size[0], self.image_size[1]) # (batch_size, num_patches + 1, C, self.image_size[0], self.image_size[1])

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

        all_box_embeddings = torch.stack([outputs["mu_min"], outputs["mu_max"]], dim=2) # (batch_size, num_patches + 1, 2, self.embed_dim)
        all_box_tensors = BoxTensor(all_box_embeddings)
        patch_box_tensors = BoxTensor(all_box_embeddings[:, :-1, :, :])
        image_box_tensor = BoxTensor(all_box_embeddings[:, -1, :, :])

        patch_intersection_box = patch_box_tensors[:, 0]
        for idx in range(1, num_patches):
            patch_intersection_box = self.box_intersection(patch_intersection_box, patch_box_tensors[:, idx])

        image_box_volume = self.box_volume(image_box_tensor)
        image_patch_intersection_volume = self.box_volume(self.box_intersection(image_box_tensor, patch_intersection_box))

        reconstruction_loss = F.mse_loss(outputs["reconstructed_images"], outputs["images"]) * self.loss_weights['reconstruction_loss']
        inclusion_loss = (image_box_volume - image_patch_intersection_volume).mean() * self.loss_weights['inclusion_loss']
        box_volume_regularization_loss = self.box_volume_regularizer(all_box_tensors).mean() * self.loss_weights['box_volume_regularization_loss']

        total_loss = reconstruction_loss + inclusion_loss + box_volume_regularization_loss
        
        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "inclusion_loss": inclusion_loss,
            "box_volume_regularization_loss": box_volume_regularization_loss,
            "total_loss": total_loss,
        }
        self.log_dict(loss_dict, prog_bar=True, on_epoch=True)

        return total_loss

    
    def configure_optimizers(self):

        if self.config['trainer']['optimizer']['type'] == 'Adam':
            return torch.optim.Adam(self.parameters(), **self.config['trainer']['optimizer']['config'])
        else:
            raise ValueError(f"Optimizer type {self.config['trainer']['optimizer']['type']} not implemented.")


    def divide_image_into_patches(self, images):
        B, C, H, W = images.shape
        
        grid_h, grid_w = self.grid_size
        patch_h = H // grid_h
        patch_w = W // grid_w

        if H % grid_h != 0 or W % grid_w != 0:
            raise ValueError("Image dimensions must be divisible by grid size.")

        patches = images.view(B, C, grid_h, patch_h, grid_w, patch_w) # (B, C, grid_h, patch_h, grid_w, patch_w)
        patches = patches.permute(0, 2, 4, 1, 3, 5) # (B, grid_h, grid_w, C, patch_h, patch_w)
        patches = patches.contiguous()
        
        return patches