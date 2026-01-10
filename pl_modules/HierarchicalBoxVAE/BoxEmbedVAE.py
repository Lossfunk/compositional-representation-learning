import torch
from torch import nn
from torch.nn import functional as F

from .box_utils import BoxDistribution

class BoxEmbedVAE(nn.Module):
    def __init__(self, vae_config, embed_dim, input_resolution=(64, 64), beta_scale=0.4, beta_pre_init=-3.0, beta_activation="sigmoid"):
        super(BoxEmbedVAE, self).__init__()

        self.vae_config = vae_config
        self.embed_dim = embed_dim
        self.hidden_dims = vae_config["hidden_dims"]
        self.beta_scale = beta_scale
        self.beta_pre_init = beta_pre_init
        self.beta_activation = beta_activation
        
        # --- Encoder ---
        encoder_modules = []
        in_channels = 3
        for h_dim in self.hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)
        
        self.final_spatial_h = input_resolution[0] // (2 ** len(self.hidden_dims))
        self.final_spatial_w = input_resolution[1] // (2 ** len(self.hidden_dims))

        self.flatten_dim = self.hidden_dims[-1] * self.final_spatial_h * self.final_spatial_w
        
        self.fc_mu_min = nn.Linear(self.flatten_dim, embed_dim)
        self.fc_mu_delta = nn.Linear(self.flatten_dim, embed_dim)
        
        self.fc_beta_min = nn.Linear(self.flatten_dim, embed_dim)
        self.fc_beta_max = nn.Linear(self.flatten_dim, embed_dim)

        nn.init.constant_(self.fc_beta_min.bias, self.beta_pre_init)
        nn.init.constant_(self.fc_beta_max.bias, self.beta_pre_init)

        self.decoder_input = nn.Linear(embed_dim * 2, self.flatten_dim)


        # --- Decoder ---
        decoder_modules = []
        decoder_dims = list(reversed(self.hidden_dims))
        
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

        batch_mu_min = torch.sigmoid(self.fc_mu_min(bottleneck_flat))
        batch_mu_delta = F.softplus(self.fc_mu_delta(bottleneck_flat))
        batch_mu_max = torch.clamp(batch_mu_min + batch_mu_delta, max = 2.0)

        if self.beta_activation == "sigmoid":
            batch_beta_min = torch.sigmoid(self.fc_beta_min(bottleneck_flat)) * self.beta_scale + 1e-6
            batch_beta_max = torch.sigmoid(self.fc_beta_max(bottleneck_flat)) * self.beta_scale + 1e-6
        elif self.beta_activation == "softplus":
            batch_beta_min = F.softplus(self.fc_beta_min(bottleneck_flat)) + 1e-6
            batch_beta_max = F.softplus(self.fc_beta_max(bottleneck_flat)) + 1e-6

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

        batch_box_dists = BoxDistribution(batch_mu_min, batch_mu_max, batch_beta_min, batch_beta_max)
        batch_samples_min, batch_samples_max = batch_box_dists.sample()
        batch_samples = torch.concat([batch_samples_min, batch_samples_max], dim = -1)
        batch_reconstructions = self.decode(batch_samples)

        return {
            'reconstructions': batch_reconstructions,
            'box_distributions': batch_box_dists,
            'samples': batch_samples
        }