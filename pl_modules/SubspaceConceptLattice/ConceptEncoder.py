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
        self.mapping_type = config["mapping_type"]
        self.heads = config["heads"]
        
        # Learnable Queries (Q) for attr and obj subspaces. These represent the basis vectors for the subspaces. 
        self.query_attr = nn.Parameter(torch.randn(self.n_attr, self.embed_dim))
        self.query_obj = nn.Parameter(torch.randn(self.n_obj, self.embed_dim))
        
        # Shared Multi-Head Attention for pooling
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.heads, batch_first=True)
        
        # Mappings between the ambient and embeddings spaces
        if self.mapping_type == "linear":
            self.attr_enc = nn.Linear(self.embed_dim, self.ambient_dim)
            self.attr_dec = nn.Linear(self.ambient_dim, self.embed_dim, bias=False)
            self.obj_enc = nn.Linear(self.embed_dim, self.ambient_dim)
        elif self.mapping_type == "mlp":
            self.attr_enc = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.ambient_dim)
            )
            self.attr_dec = nn.Sequential(
                nn.Linear(self.ambient_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
            self.obj_enc = nn.Sequential(
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
        
        # Flatten the subset and patches into a single continuous sequence of tokens
        H = R_subset.view(B, S * P, E) # (B, S*P, E)
        
        # ==========================================
        # Pathway A: The Attribute Subspace (S^attr)
        # ==========================================
        # Expand queries for the batch
        q_attr = self.query_attr.unsqueeze(0).expand(B, -1, -1) # (B, n_attr, embed_dim)
        # Attention pools the variable-length sequence H into fixed n_attr vectors
        x_attr, _ = self.attn(query=q_attr, key=H, value=H) # (B, n_attr, embed_dim)
        # Map to ambient dimension D to get the basis vectors X
        X_attr = self.attr_enc(x_attr) # (B, n_attr, ambient_dim)
        # Compute smooth projection operator \tilde{P}
        P_attr = ridge_projector(X_attr, lbd=self.lbd) # (B, ambient_dim, ambient_dim)
        
        # ==========================================
        # Pathway B: The Object Subspace (S^obj)
        # ==========================================
        q_obj = self.query_obj.unsqueeze(0).expand(B, -1, -1) # (B, n_obj, embed_dim)
        x_obj, _ = self.attn(query=q_obj, key=H, value=H) # (B, n_obj, embed_dim)
        X_obj = self.obj_enc(x_obj) # (B, n_obj, ambient_dim)
        P_obj = ridge_projector(X_obj, lbd=self.lbd) # (B, ambient_dim, ambient_dim)
        
        return X_attr, X_obj, P_attr, P_obj