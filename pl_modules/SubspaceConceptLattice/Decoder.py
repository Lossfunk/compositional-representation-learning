import torch
from torch import nn

class ViTDecoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.out_channels = config["image_channels"]
        self.embed_dim = config["embed_dim"]
        self.depth = config["depth"]
        self.heads = config["heads"]
        self.mlp_ratio = config["mlp_ratio"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        out_dim = self.out_channels * (self.patch_size ** 2)
        
        # 1. Learnable spatial queries (one for each patch location)
        self.spatial_queries = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))
        
        # 2. Transformer Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.heads,
            dim_feedforward=int(self.embed_dim * self.mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.depth)
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # 3. Project decoded tokens back to pixel patches
        self.head = nn.ConvTranspose2d(self.embed_dim, self.out_channels, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, concept_vectors):
        # concept_vectors shape: (B, N, embed_dim)
        B = concept_vectors.shape[0]
        
        # Expand spatial queries for the batch: (B, num_patches, embed_dim)
        queries = self.spatial_queries.expand(B, -1, -1)
        
        # Cross-attention: Queries ask the Concept Vectors for information
        # tgt = queries, memory = concept_vectors
        x = self.transformer(tgt=queries, memory=concept_vectors)
        x = self.norm(x)
        
        # Reshape to spatial feature map: (B, embed_dim, H/P, W/P)
        H_P = self.image_size // self.patch_size
        W_P = self.image_size // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_P, W_P)
        
        # Project to pixels using ConvTranspose2d: (B, C, H, W)
        x = self.head(x)
        
        return x