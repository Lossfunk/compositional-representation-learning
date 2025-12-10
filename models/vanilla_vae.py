import torch
from torch import nn


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
