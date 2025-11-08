import torch
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.half_embed_dim = embed_dim // 2

    def generate_sinusoidal_1d(self, positions):
        batch_size, N = positions.shape[0], positions.shape[1]
        positions = positions.reshape(-1, 1)  # (batch_size * N, 1)

        i = torch.arange(0, self.half_embed_dim, 2, dtype=torch.float32, device=positions.device)
        denominator = torch.pow(
            10000, i / self.half_embed_dim
        )  # shape: (half_embed_dim // 2,) # (batch_size * N, half_embed_dim // 2)
        denominator = positions / denominator

        pos_embeddings = torch.zeros(
            batch_size * N,
            self.half_embed_dim,
            device=positions.device,
            dtype=torch.float32,
        )
        pos_embeddings[:, ::2] = torch.sin(denominator)
        pos_embeddings[:, 1::2] = torch.cos(denominator)

        return pos_embeddings.reshape(batch_size, N, self.half_embed_dim)

    def forward(self, coordinates: torch.Tensor):  # (batch_size, N, 2)
        y_embeddings = self.generate_sinusoidal_1d(
            coordinates[:, :, 0]
        )  # (batch_size, N, half_embed_dim)
        x_embeddings = self.generate_sinusoidal_1d(
            coordinates[:, :, 1]
        )  # (batch_size, N, half_embed_dim)
        return torch.cat([y_embeddings, x_embeddings], dim=-1)  # (batch_size, N, embed_dim)
