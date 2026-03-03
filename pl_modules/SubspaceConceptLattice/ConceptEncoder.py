import torch
from torch import nn

from .concept_utils import ridge_projector

class ConceptEncoder(nn.Module):
    def __init__(
        self, 
        config
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.n_attr = config["n_attr"]
        self.n_obj = config["n_obj"]
        self.ambient_dim = config["ambient_dim"]
        self.lbd = config["lbd"]
        self.heads = config["heads"]
        
        # 1. Learnable Queries (Q) for both elements of the formal concept 
        self.query_attr = nn.Parameter(torch.randn(self.n_attr, self.embed_dim))
        self.query_obj = nn.Parameter(torch.randn(self.n_obj, self.embed_dim))
        
        # 2. Shared Multi-Head Attention for pooling [cite: 145, 146]
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.heads, batch_first=True)
        
        # 3. Non-linear mapping (MLP) to the ambient subspace dimension [cite: 149, 150]
        self.mlp_attr = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.ambient_dim)
        )
        
        # self.mlp_attr_decoder = nn.Sequential(
        #     nn.Linear(self.ambient_dim, self.embed_dim),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(self.embed_dim),
        #     nn.Linear(self.embed_dim, self.embed_dim)
        # )
        self.mlp_attr_decoder = nn.Linear(self.ambient_dim, self.embed_dim, bias=False)
        
        self.mlp_obj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.ambient_dim)
        )

    def forward(self, R_subset):
        """
        Args:
            R_subset: Tensor of shape (Batch, Subset_Size, Num_Patches, Embed_Dim)
                      representing a subset of images encoded by the ViT.
        """
        B, S, P, E = R_subset.shape
        
        # Flatten the subset and patches into a single continuous sequence of tokens[cite: 142].
        # Shape becomes: (Batch, Subset_Size * Num_Patches, Embed_Dim)
        H = R_subset.view(B, S * P, E)
        
        # ==========================================
        # Pathway A: The Attribute Subspace (S^attr)
        # ==========================================
        # Expand queries for the batch
        q_attr = self.query_attr.unsqueeze(0).expand(B, -1, -1)
        
        # Attention pools the variable-length sequence H into fixed n_attr vectors [cite: 146, 148]
        x_attr, _ = self.attn(query=q_attr, key=H, value=H)
        
        # Map to ambient dimension D to get the basis vectors X [cite: 149, 150]
        X_attr = self.mlp_attr(x_attr) # Shape: (B, n_attr, ambient_dim)
        
        # Compute smooth projection operator \tilde{P} [cite: 152]
        P_attr = ridge_projector(X_attr, lbd=self.lbd) # Shape: (B, ambient_dim, ambient_dim)
        
        # ==========================================
        # Pathway B: The Object Subspace (S^obj)
        # ==========================================
        q_obj = self.query_obj.unsqueeze(0).expand(B, -1, -1)
        x_obj, _ = self.attn(query=q_obj, key=H, value=H)
        X_obj = self.mlp_obj(x_obj)
        P_obj = ridge_projector(X_obj, lbd=self.lbd)
        
        # We return the basis vectors X_attr for the Decoder to use, 
        # and the projectors P_attr / P_obj for the loss functions.
        return X_attr, X_obj, P_attr, P_obj