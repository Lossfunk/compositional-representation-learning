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
        
        # Learnable spatial queries (one for each patch location)
        self.spatial_queries = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))
        
        # Transformer Decoder layers
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
        
        # Project decoded tokens back to pixel patches
        self.head = nn.ConvTranspose2d(self.embed_dim, self.out_channels, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, concept_vectors):
        # Concept vectors represent the basis vectors for the attribute subspace
        B = concept_vectors.shape[0] # (B, N, embed_dim)
        
        # Expand spatial queries for the batch
        queries = self.spatial_queries.expand(B, -1, -1) # (B, num_patches, embed_dim)
        
        # Cross-attention: Queries ask the Concept Vectors for information
        x = self.transformer(tgt=queries, memory=concept_vectors) # (B, num_patches, embed_dim)
        x = self.norm(x) # (B, num_patches, embed_dim)
        
        # Reshape to spatial feature map
        H_P = self.image_size // self.patch_size
        W_P = self.image_size // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_P, W_P) # (B, embed_dim, H/P, W/P)
        
        # Project to pixel space using ConvTranspose2d
        x = self.head(x) # (B, C, H, W)
        
        return x


class VQDecoder(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.config = config

        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.out_channels = config["image_channels"]
        self.embed_dim = config["embed_dim"]
        self.attr_ambient_dim = config["attr_ambient_dim"]
        self.depth = config["depth"]
        self.heads = config["heads"]
        self.mlp_ratio = config["mlp_ratio"]
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.spatial_queries = nn.Parameter(torch.randn(1, self.num_patches, self.attr_ambient_dim))

        self.query_proj = nn.Linear(self.ambient_dim, self.embed_dim, bias=False)

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

        self.head = nn.ConvTranspose2d(
            self.embed_dim, 
            self.out_channels, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

    def forward(self, P_attr):
        B = P_attr.shape[0]

        queries = self.spatial_queries.expand(B, -1, -1) # (B, num_patches, attr_ambient_dim)
        filtered_queries = torch.bmm(queries, P_attr) # (B, num_patches, attr_ambient_dim)
        x = self.query_proj(filtered_queries) # (B, num_patches, embed_dim)

        x = self.transformer(x) # (B, num_patches, embed_dim)
        x = self.norm(x)

        H_P = self.image_size // self.patch_size
        W_P = self.image_size // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_P, W_P) # (B, embed_dim, H/P, W/P)

        x = self.head(x) # (B, C, H, W)

        return x




