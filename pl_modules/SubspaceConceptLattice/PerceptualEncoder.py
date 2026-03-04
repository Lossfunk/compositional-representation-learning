import torch
from torch import nn

class ViTEncoder(nn.Module):
    def __init__(
        self, 
        config,
    ):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.in_channels = config["image_channels"]
        self.embed_dim = config["embed_dim"]
        self.depth = config["depth"]
        self.heads = config["heads"]
        self.mlp_ratio = config["mlp_ratio"]
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Patchify and project to embedding dimension
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))
        
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=self.heads, 
            dim_feedforward=int(self.embed_dim * self.mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        B = x.shape[0]
        
        # Patchify -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x).flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed # (B, num_patches, embed_dim)
        
        # Pass through transformer
        x = self.transformer(x) # (B, num_patches, embed_dim)
        x = self.norm(x) # (B, num_patches, embed_dim)
        
        return x