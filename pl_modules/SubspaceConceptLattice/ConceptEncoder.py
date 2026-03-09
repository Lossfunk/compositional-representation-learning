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
                # nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.ambient_dim, bias=False)
            )
            self.attr_dec = nn.Sequential(
                nn.Linear(self.ambient_dim, self.embed_dim),
                nn.LeakyReLU(),
                # nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            )
            self.obj_enc = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LeakyReLU(),
                # nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.ambient_dim, bias=False)
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


class VQSlotHead(nn.Module):
    def __init__(self, embed_dim, capacity):
        super().__init__()
        self.embed_dim = embed_dim
        self.capacity = capacity

        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, capacity, bias=False)
        )

        codebook = torch.cat([
            torch.zeros(1, capacity),
            torch.eye(capacity)
        ], dim=0)
        self.register_buffer("codebook", codebook)

    def forward(self, x):
        B, E = x.shape

        logits = self.proj(x) # (B, capacity)
        probs = torch.sigmoid(logits) # (B, capacity)

        distances = torch.cdist(x, self.codebook) # (B, capacity + 1)
        nearest_indices = torch.argmin(distances, dim=-1) # (B,)
        discrete_v = self.codebook[nearest_indices] # (B, capacity)

        v_out = discrete_v.detach() - probs.detach() + probs
        commitment_loss = F.mse_loss(probs, discrete_v.detach()) 

        return v_out, commitment_loss


class VQObjectHead(nn.Module):
    def __init__(self, embed_dim, n_obj):
        super().__init__()
        self.n_obj = n_obj

        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_obj, bias=False)
        )

    def forward(self, x):
        B, E = x.shape

        logits = self.proj(x) # (B, n_obj)
        probs = torch.sigmoid(logits) # (B, n_obj)

        discrete_v = (probs > 0.5).float()
        v_out = discrete_v.detach() - probs.detach() + probs

        return v_out


class VQConceptEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
       
        self.embed_dim = config["embed_dim"]
        self.attr_ambient_dim = config["attr_ambient_dim"]
        self.obj_ambient_dim = config["obj_ambient_dim"]
        self.attr_structure = config["attr_structure"]
        self.n_obj = config["n_obj"]
        self.n_attr_slots = len(self.attr_structure)
        
        self.lbd = config["lbd"]
        self.mapping_type = config["mapping_type"]
        self.heads = config["heads"]

        # Learnable Queries
        self.query_attr = nn.Parameter(torch.randn(self.n_attr_slots, self.embed_dim))
        self.query_obj = nn.Parameter(torch.randn(1, self.embed_dim))

        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.heads, batch_first=True)

        # Output Heads
        self.attr_heads = nn.ModuleList([
            VQSlotHead(self.embed_dim, capacity) for capacity in self.attr_structure
        ])
        self.obj_head = VQObjectHead(self.embed_dim, self.n_obj)

    def forward(self, R_subset):
        B, S, P, E = R_subset.shape
        H = R_subset.view(B, S * P, E)

        # Attribute Pathway
        q_attr = self.query_attr.unsqueeze(0).expand(B, -1, -1) # (B, n_attr_slots, embed_dim)
        x_attr, _ = self.attn(query=q_attr, key=H, value=H) # (B, n_attr_slots, embed_dim)
        
        slot_vectors = []
        attr_commitment_loss = 0
        for i, slot_head in enumerate(self.attr_heads):
            v_out, slot_commitment_loss = slot_head(x_attr[:, i, :])
            slot_vectors.append(v_out) # (B, capacity)
            attr_commitment_loss += slot_commitment_loss

        X_attr = torch.stack(slot_vectors, dim=-1) # (B, attr_ambient_dim)
        attr_commitment_loss /= self.n_attr_slots

        # Object Pathway
        q_obj = self.query_obj.unsqueeze(0).expand(B, -1, -1) # (B, 1, embed_dim)
        x_obj, _ = self.attn(query=q_obj, key=H, value=H) # (B, 1, embed_dim)
        X_obj = self.obj_head(x_obj.squeeze(1)) # (B, n_obj)

        return X_attr, X_obj, attr_commitment_loss

        