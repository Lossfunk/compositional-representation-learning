import torch
from torch import nn
import torch.nn.functional as F

from .box_utils import BoxEmbeddingDistribution


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
            nn.Sigmoid(),
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


class BoxEmbedVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dims, input_resolution=(64, 64)):
        super(BoxEmbedVAE, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # --- Encoder ---
        encoder_modules = []
        in_channels = 3
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)
        
        self.final_spatial_h = input_resolution[0] // (2 ** len(hidden_dims))
        self.final_spatial_w = input_resolution[1] // (2 ** len(hidden_dims))

        self.flatten_dim = hidden_dims[-1] * self.final_spatial_h * self.final_spatial_w
        
        self.fc_mu_min = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_mu_delta = nn.Linear(self.flatten_dim, latent_dim)
        
        self.fc_beta_min = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_beta_max = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim * 2, self.flatten_dim)

        decoder_modules = []
        decoder_dims = list(reversed(hidden_dims))
        
        for i in range(len(decoder_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(decoder_dims[i], decoder_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(decoder_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(decoder_dims[-1], decoder_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )


    def encode(self, input):
        bottleneck = self.encoder(input)
        bottleneck_flat = torch.flatten(bottleneck, start_dim=1)

        batch_mu_min = self.fc_mu_min(bottleneck_flat)
        batch_mu_delta = self.fc_mu_delta(bottleneck_flat)
        batch_mu_max = batch_mu_min + F.softplus(batch_mu_delta)

        batch_beta_min = F.softplus(self.fc_beta_min(bottleneck_flat))
        batch_beta_max = F.softplus(self.fc_beta_max(bottleneck_flat))

        return batch_mu_min, batch_mu_max, batch_beta_min, batch_beta_max

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.final_spatial_h, self.final_spatial_w)
        result = self.decoder(result)
        result = self.final_layer(result)
        
        return result

    def forward(self, input, **kwargs):
        batch_size = input.shape[0]
        batch_mu_min, batch_mu_max, batch_beta_min, batch_beta_max = self.encode(input)

        # batch_box_dists = []
        # batch_samples = []
        # for datapoint_idx in batch_size:
        #     datapoint_box_dist = BoxEmbeddingDistribution(batch_mu_min[datapoint_idx], batch_mu_max[datapoint_idx], batch_beta_min[datapoint_idx], batch_beta_max[datapoint_idx])
        #     datapoint_sample = datapoint_box_dist.sample()
            
        #     batch_box_dists.append(datapoint_box_dist)
        #     batch_samples.append(datapoint_sample)

        # batch_samples = torch.stack(batch_samples, dim = 0)
        # reconstruction = self.decode(batch_samples)

        batch_box_dists = BoxEmbeddingDistribution(batch_mu_min, batch_mu_max, batch_beta_min, batch_beta_max)
        batch_samples_min, batch_samples_max = batch_box_dists.sample()
        batch_samples = torch.concat([batch_samples_min, batch_samples_max], dim = -1)
        batch_reconstructions = self.decode(batch_samples)

        return {
            'reconstructions': batch_reconstructions,
            'box_distributions': batch_box_dists,
            'samples': batch_samples
        }