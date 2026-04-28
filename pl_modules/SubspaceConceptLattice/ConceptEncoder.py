import torch
from torch import nn
from torch.nn import functional as F

from .concept_utils import ridge_projector, gumbel_sigmoid


class RepresentationCombiner(nn.Module):
    """
    Combines 2-N per-image representations into a single combined representation
    before concept encoding. Uses cross-attention: a set of learnable combination
    queries attend to the concatenated patch tokens from all images in the subset,
    producing a single fixed-size representation (1, num_patches, embed_dim).

    For singletons (k=1), this is a pass-through — the single representation
    is returned unchanged.
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.num_patches = config["num_patches"]
        self.heads = config["heads"]
        self.depth = config.get("combiner_depth", 2)

        # Learnable queries: one per output patch position
        # These will cross-attend to all input patch tokens from all images
        self.combination_queries = nn.Parameter(
            torch.randn(1, self.num_patches, self.embed_dim)
        )

        # Cross-attention layers to combine multi-image representations
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.heads,
                batch_first=True
            )
            for _ in range(self.depth)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim)
            for _ in range(self.depth)
        ])

    def forward(self, R_subset):
        """
        Args:
            R_subset: (num_combinations, k, num_patches, embed_dim)
                k images per combination, each with num_patches patch tokens.
        Returns:
            R_combined: (num_combinations, 1, num_patches, embed_dim)
                A single combined representation per combination.
        """
        NC, k, P, E = R_subset.shape

        # For singletons, pass through unchanged
        if k == 1:
            return R_subset

        # Flatten all k images' patches into one long sequence
        # (NC, k*P, E)
        kv = R_subset.view(NC, k * P, E)

        # Expand queries for the batch
        queries = self.combination_queries.expand(NC, -1, -1)  # (NC, P, E)

        # Apply cross-attention layers
        x = queries
        for cross_attn, norm in zip(self.cross_attn_layers, self.norms):
            residual = x
            x = norm(x)
            x, _ = cross_attn(query=x, key=kv, value=kv)
            x = x + residual

        # Reshape to match expected format: (NC, 1, P, E)
        return x.unsqueeze(1)


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
        self.clamp_X = config["clamp_X"]
        self.attr_force_orthogonal_basis = config["attr_force_orthogonal_basis"]
        self.subspace_gumbel_sigmoid = config["subspace_gumbel_sigmoid"]
        self.fillers_per_slot = config.get("fillers_per_slot", None)

        # Compute per-slot dimension allocation for the orthogonal basis mask
        if config.get("dims_per_slot", None) is not None:
            self.dims_per_slot = config["dims_per_slot"]
        elif self.attr_force_orthogonal_basis:
            # Equal allocation by default
            base = self.ambient_dim // self.n_attr
            remainder = self.ambient_dim % self.n_attr
            self.dims_per_slot = [base + (1 if i < remainder else 0) for i in range(self.n_attr)]
        else:
            self.dims_per_slot = [self.ambient_dim // self.n_attr] * self.n_attr
        
        # Annealing Configs
        self.gumbel_sigmoid_annealing = config.get("gumbel_sigmoid_annealing", False)
        self.anneal_start_epoch = config.get("gumbel_sigmoid_annealing_start_epoch", 0)
        self.anneal_end_epoch = config.get("gumbel_sigmoid_annealing_end_epoch", 0)
        self.start_temp = config.get("gumbel_sigmoid_annealing_start_temp", 1.0)
        self.end_temp = config.get("gumbel_sigmoid_annealing_end_temp", 0.1)
        self.gumbel_sigmoid_hard = config.get("gumbel_sigmoid_hard", False)
        
        # Learnable Queries (Q) for attr and obj subspaces. These represent the basis vectors for the subspaces. 
        self.query_attr = nn.Parameter(torch.randn(self.n_attr, self.embed_dim))
        self.query_obj = nn.Parameter(torch.randn(self.n_obj, self.embed_dim))
        
        # Shared transformer for pooling
        self.encoder_depth = config.get("encoder_depth", 1)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.heads, batch_first=True)
            for _ in range(self.encoder_depth)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim)
            for _ in range(self.encoder_depth)
        ])
        
        # Mappings between the ambient and embeddings spaces
        if self.mapping_type == "linear":
            self.attr_enc = nn.Linear(self.embed_dim, self.ambient_dim)
            self.attr_dec = nn.Linear(self.ambient_dim, self.embed_dim, bias=False)
            self.obj_enc = nn.Linear(self.embed_dim, self.ambient_dim)
        elif self.mapping_type == "mlp":
            self.attr_enc = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                # nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.ambient_dim)
            )
            self.attr_dec = nn.Sequential(
                nn.Linear(self.ambient_dim, self.embed_dim),
                nn.ReLU(),
                # nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
            self.obj_enc = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                # nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.ambient_dim)
            )

    def get_temperature(self, current_epoch):
        if not self.gumbel_sigmoid_annealing or current_epoch is None:
            return sum((self.start_temp, self.end_temp)) / 2.0 if self.gumbel_sigmoid_annealing else 0.25
        
        if current_epoch < self.anneal_start_epoch:
            return self.start_temp
        elif current_epoch >= self.anneal_end_epoch:
            return self.end_temp
        else:
            progress = (current_epoch - self.anneal_start_epoch) / max(1, self.anneal_end_epoch - self.anneal_start_epoch)
            return self.start_temp + progress * (self.end_temp - self.start_temp)

    def forward(self, R_subset, current_epoch=None):
        """
        Args:
            R_subset: Tensor of shape (Batch, Subset_Size, Num_Patches, Embed_Dim)
                      representing a subset of images encoded by the ViT.
            current_epoch: The current training epoch (optional, used for annealing)
        """
        B, S, P, E = R_subset.shape
        
        # Flatten the subset and patches into a single continuous sequence of tokens
        H = R_subset.view(B, S * P, E) # (B, S*P, E)
        
        # ==========================================
        # Pathway A: The Attribute Subspace (S^attr)
        # ==========================================
        # Expand queries for the batch
        q_attr = self.query_attr.unsqueeze(0).expand(B, -1, -1) # (B, n_attr, embed_dim)
        # Cross-attention stack pools the variable-length sequence H into fixed n_attr vectors
        x_attr = q_attr
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            residual = x_attr
            x_attr = norm(x_attr)
            x_attr, _ = attn(query=x_attr, key=H, value=H)
            x_attr = x_attr + residual
        # Map to ambient dimension D to get the basis vectors X
        X_attr = self.attr_enc(x_attr) # (B, n_attr, ambient_dim)

        if self.subspace_gumbel_sigmoid:
            tau = self.get_temperature(current_epoch)
            X_attr = gumbel_sigmoid(X_attr, tau=tau, hard=self.gumbel_sigmoid_hard)

        if self.attr_force_orthogonal_basis:
            mask = torch.zeros((self.n_attr, self.ambient_dim), device=X_attr.device)
            offset = 0
            for i in range(self.n_attr):
                d = self.dims_per_slot[i]
                mask[i, offset:offset + d] = 1.0
                offset += d

            X_attr = X_attr * mask.unsqueeze(0)
        
        if self.clamp_X:
            X_attr = torch.clamp(X_attr, min=0, max=1)
        # Compute smooth projection operator \tilde{P}
        P_attr = ridge_projector(X_attr, lbd=self.lbd) # (B, ambient_dim, ambient_dim)
        
        # ==========================================
        # Pathway B: The Object Subspace (S^obj)
        # ==========================================
        q_obj = self.query_obj.unsqueeze(0).expand(B, -1, -1) # (B, n_obj, embed_dim)
        x_obj = q_obj
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            residual = x_obj
            x_obj = norm(x_obj)
            x_obj, _ = attn(query=x_obj, key=H, value=H)
            x_obj = x_obj + residual
        X_obj = self.obj_enc(x_obj) # (B, n_obj, ambient_dim)
        if self.clamp_X:
            X_obj = torch.clamp(X_obj, min=0, max=1)
        P_obj = ridge_projector(X_obj, lbd=self.lbd) # (B, ambient_dim, ambient_dim)
        
        return X_attr, X_obj, P_attr, P_obj


class MemoryConceptEncoder(nn.Module):
    """
    Memory-based concept encoder with persistent attribute and object codebooks.

    Takes R_subset (B, S, P, E) — cardinality-wise combinations just like
    ConceptEncoder.  Produces raw X_attr and X_obj in ambient space, then
    resolves each against memory using distance-based scoring.

    Attribute resolution:
        For each slot, candidates = {zero_vec, filler_0, ..., filler_K}.
        The raw proposal is compared against all Cartesian-product combinations
        of these per-slot candidates (including the zero vector).
        Index 0 = zero vector (absent), indices 1..K = memory fillers.

    Object resolution:
        Object memory = (n_obj, ambient_dim) rank-1 subspace vectors.
        Candidates = all 2^n_obj non-empty subsets (unions of obj vectors).
        The raw proposal's projector is compared via inclusion against each
        union's projector.

    Resolution methods:
        "cosine"    — cosine similarity in full ambient space
        "inclusion" — subspace inclusion scoring via ridge projectors
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.ambient_dim = config["ambient_dim"]
        self.n_attr = config["n_attr"]
        self.n_obj = config["n_obj"]
        self.lbd = config["lbd"]
        self.heads = config["heads"]
        self.fillers_per_slot = config["fillers_per_slot"]  # e.g. [2, 2]

        # Resolution method: "argmax" (hard argmax + STE) or "gumbel_softmax"
        self.resolution_method = config.get("resolution_method", "argmax")
        self.resolution_tau = config.get("resolution_tau", 1.0)

        # Commitment loss asymmetry: weight for codebook vs encoder terms.
        # codebook_weight controls how fast memory moves toward proposals.
        # encoder_weight controls how fast proposals move toward memory.
        # Default: both 1.0 (symmetric, original VQ-VAE).
        # For stable memory: lower codebook_weight (e.g., 0.25).
        self.commitment_codebook_weight = config.get("commitment_codebook_weight", 1.0)
        self.commitment_encoder_weight = config.get("commitment_encoder_weight", 1.0)

        # Per-slot dimension allocation
        if config.get("dims_per_slot") is not None:
            self.dims_per_slot = config["dims_per_slot"]
        else:
            base = self.ambient_dim // self.n_attr
            remainder = self.ambient_dim % self.n_attr
            self.dims_per_slot = [base + (1 if i < remainder else 0)
                                  for i in range(self.n_attr)]

        assert sum(self.dims_per_slot) == self.ambient_dim

        # ── Attribute Memory ────────────────────────────────────────────
        # Per-slot codebook.  Entry vectors live in the slot's dim-slice.
        self.attr_memory = nn.ParameterList([
            nn.Parameter(torch.randn(n_fillers, d) * 0.1)
            for n_fillers, d in zip(self.fillers_per_slot, self.dims_per_slot)
        ])

        # ── Object Memory ──────────────────────────────────────────────
        # n_obj subspaces, each defined by n_obj basis vectors in ambient_dim.
        # Target: each entry is rank-1 (all rows identical), entries are
        # mutually orthogonal. Pick n_obj orthonormal directions, assign
        # one per entry, and tile it across all n_obj rows.
        Q, _ = torch.linalg.qr(torch.randn(self.ambient_dim, self.n_obj))
        directions = Q.t()  # (n_obj, ambient_dim), orthonormal rows
        obj_mem_init = directions.unsqueeze(1).expand(
            self.n_obj, self.n_obj, self.ambient_dim
        ).clone()  # (n_obj, n_obj, ambient_dim) — each entry has identical rows
        self.obj_memory = nn.Parameter(obj_mem_init)

        # ── Proposal Network ───────────────────────────────────────────
        # Attention hierarchy (each level requires the previous):
        #   Level 0: shared TransformerDecoder for attr+obj queries (simplest)
        #   Level 1: separate_obj_attr_attn — separate decoders for attr vs obj
        #   Level 2: separate_slot_attn — separate decoders per attr slot
        self.separate_obj_attr_attn = config.get("separate_obj_attr_attn", False)
        self.separate_slot_attn = config.get("separate_slot_attn", False)
        if self.separate_slot_attn:
            assert self.separate_obj_attr_attn, \
                "separate_slot_attn requires separate_obj_attr_attn=True"

        attn_depth = config.get("attn_depth", 1)
        attn_heads = config.get("attn_heads", self.heads)
        attn_mlp_ratio = config.get("attn_mlp_ratio", 4.0)

        def _make_decoder(n_layers):
            layer = nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=attn_heads,
                dim_feedforward=int(self.embed_dim * attn_mlp_ratio),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerDecoder(layer, num_layers=n_layers)

        # Learnable queries
        self.query_attr = nn.Parameter(torch.randn(self.n_attr, self.embed_dim))
        self.query_obj = nn.Parameter(torch.randn(self.n_obj, self.embed_dim))

        if not self.separate_obj_attr_attn:
            # Level 0: single shared decoder, attr+obj queries concatenated
            self.shared_decoder = _make_decoder(attn_depth)
            self.shared_decoder_norm = nn.LayerNorm(self.embed_dim)
        elif not self.separate_slot_attn:
            # Level 1: separate decoders for attr vs obj
            self.attr_decoder = _make_decoder(attn_depth)
            self.attr_decoder_norm = nn.LayerNorm(self.embed_dim)
            self.obj_decoder = _make_decoder(attn_depth)
            self.obj_decoder_norm = nn.LayerNorm(self.embed_dim)
        else:
            # Level 2: per-slot decoders for attr + separate obj decoder
            self.slot_decoders = nn.ModuleList([
                _make_decoder(attn_depth) for _ in range(self.n_attr)
            ])
            self.slot_decoder_norms = nn.ModuleList([
                nn.LayerNorm(self.embed_dim) for _ in range(self.n_attr)
            ])
            self.obj_decoder = _make_decoder(attn_depth)
            self.obj_decoder_norm = nn.LayerNorm(self.embed_dim)

        # Per-slot projection: embed_dim → slot's ambient dims
        self.slot_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, d),
            )
            for d in self.dims_per_slot
        ])

        # Object projection: embed_dim → ambient_dim
        self.obj_projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.ambient_dim),
        )

        # ── Reconstruction pathway ─────────────────────────────────────
        self.attr_dec = nn.Sequential(
            nn.Linear(self.ambient_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # ── Precompute candidate structures ────────────────────────────
        # These are index tuples, rebuilt if needed; the actual vectors
        # come from memory at forward time.
        self._build_attr_candidates_index()
        self._build_obj_candidates_index()
        self._build_singleton_candidates_index()

    # ------------------------------------------------------------------ #
    #  Candidate index builders
    # ------------------------------------------------------------------ #
    def _build_attr_candidates_index(self):
        """
        Build the Cartesian product index of all attr memory combinations.
        Per slot: index 0 = zero vector, indices 1..K = memory fillers.
        So slot with K fillers has K+1 options.
        Total candidates = product of (K_i + 1) for all slots.
        Also precomputes hardcoded ranks (= number of non-zero slots).
        """
        import itertools as _it
        options_per_slot = [range(k + 1) for k in self.fillers_per_slot]
        self._attr_candidate_tuples = list(_it.product(*options_per_slot))
        self.n_attr_candidates = len(self._attr_candidate_tuples)
        # Rank = number of active (non-zero) slots
        self._attr_candidate_ranks = [
            sum(1 for c in combo if c != 0)
            for combo in self._attr_candidate_tuples
        ]

    def _build_obj_candidates_index(self):
        """
        Build all non-empty subsets of object memory indices.
        For n_obj objects: 2^n_obj - 1 candidates.
        Each candidate is a tuple of object indices to union.
        Also precomputes hardcoded ranks (= number of entries in subset).
        """
        import itertools as _it
        self._obj_candidate_subsets = []
        for r in range(1, self.n_obj + 1):
            for subset in _it.combinations(range(self.n_obj), r):
                self._obj_candidate_subsets.append(subset)
        self.n_obj_candidates = len(self._obj_candidate_subsets)
        # Rank = number of base subspaces in the union
        self._obj_candidate_ranks = [len(s) for s in self._obj_candidate_subsets]

    def _build_singleton_candidates_index(self):
        """
        Build reduced candidate indices for singleton mode:
        - Attr: only filler combinations (no zero vectors per slot).
          Per slot: indices 1..K (excluding 0=zero).
          Total = product of K_i for all slots.
        - Obj: only individual object entries (no unions).
          Total = n_obj.
        """
        import itertools as _it
        options_per_slot = [range(1, k + 1) for k in self.fillers_per_slot]
        self._singleton_attr_candidate_tuples = list(_it.product(*options_per_slot))
        self.n_singleton_attr_candidates = len(self._singleton_attr_candidate_tuples)
        # All singleton attr candidates have all slots active
        self._singleton_attr_candidate_ranks = [self.n_attr] * self.n_singleton_attr_candidates

        self._singleton_obj_candidate_subsets = [(i,) for i in range(self.n_obj)]
        self.n_singleton_obj_candidates = self.n_obj
        # All singleton obj candidates have rank 1
        self._singleton_obj_candidate_ranks = [1] * self.n_singleton_obj_candidates

    # ------------------------------------------------------------------ #
    #  Build candidate vectors/projectors from memory (called at forward)
    # ------------------------------------------------------------------ #
    def _get_attr_candidate_vectors(self):
        """
        Build (n_attr_candidates, n_attr, ambient_dim) tensor of all
        attr memory combinations including zero vectors.
        """
        device = self.attr_memory[0].device
        candidates = []
        for combo in self._attr_candidate_tuples:
            basis_vecs = []
            for slot_idx, choice in enumerate(combo):
                d = self.dims_per_slot[slot_idx]
                offset_before = sum(self.dims_per_slot[:slot_idx])
                offset_after = self.ambient_dim - offset_before - d
                if choice == 0:
                    # Zero vector (absent filler)
                    vec = torch.zeros(self.ambient_dim, device=device)
                else:
                    # Memory filler (1-indexed → 0-indexed), L2-normalized
                    slot_vec = self.attr_memory[slot_idx][choice - 1]  # (d,)
                    slot_vec = F.normalize(slot_vec, p=2, dim=-1)
                    vec = F.pad(slot_vec, (offset_before, offset_after))
                basis_vecs.append(vec)
            candidates.append(torch.stack(basis_vecs))  # (n_attr, ambient_dim)
        return torch.stack(candidates)  # (n_candidates, n_attr, ambient_dim)

    def _get_singleton_attr_candidate_vectors(self):
        """
        Build (n_singleton_attr_candidates, n_attr, ambient_dim) tensor of
        attr memory combinations WITHOUT zero vectors (fillers only).
        """
        device = self.attr_memory[0].device
        candidates = []
        for combo in self._singleton_attr_candidate_tuples:
            basis_vecs = []
            for slot_idx, choice in enumerate(combo):
                d = self.dims_per_slot[slot_idx]
                offset_before = sum(self.dims_per_slot[:slot_idx])
                offset_after = self.ambient_dim - offset_before - d
                # choice is 1-indexed (1..K) — no zero option, L2-normalized
                slot_vec = self.attr_memory[slot_idx][choice - 1]  # (d,)
                slot_vec = F.normalize(slot_vec, p=2, dim=-1)
                vec = F.pad(slot_vec, (offset_before, offset_after))
                basis_vecs.append(vec)
            candidates.append(torch.stack(basis_vecs))  # (n_attr, ambient_dim)
        return torch.stack(candidates)  # (n_singleton_attr_candidates, n_attr, ambient_dim)

    def _get_singleton_obj_candidate_projectors(self):
        """
        Build projectors for individual object memory subspaces (no unions).
        Each entry is already (n_obj, ambient_dim).
        Returns:
            candidates_P: (n_obj, ambient_dim, ambient_dim)
        """
        candidates_P = []
        for subset in self._singleton_obj_candidate_subsets:
            X = self.obj_memory[subset[0]]  # (n_obj, ambient_dim)
            P = ridge_projector(X.unsqueeze(0), lbd=self.lbd).squeeze(0)
            candidates_P.append(P)
        return torch.stack(candidates_P)  # (n_obj, D, D)

    def _get_obj_candidate_projectors(self):
        """
        Build (n_obj_candidates, ambient_dim, ambient_dim) tensor of union
        projectors for each non-empty subset of object memory subspaces.

        Each obj_memory entry is (n_obj, ambient_dim). For a subset of entries,
        we concatenate their basis vectors and compute the ridge projector over
        the union.
        """
        candidates_P = []
        for subset in self._obj_candidate_subsets:
            # Concatenate basis vectors from selected entries
            parts = [self.obj_memory[i] for i in subset]  # each (n_obj, D)
            X = torch.cat(parts, dim=0)  # (|subset|*n_obj, ambient_dim)
            P = ridge_projector(X.unsqueeze(0), lbd=self.lbd).squeeze(0)
            candidates_P.append(P)
        return torch.stack(candidates_P)  # (n_obj_candidates, D, D)

    # ------------------------------------------------------------------ #
    #  Inclusion helpers
    # ------------------------------------------------------------------ #
    def _bidirectional_inclusion(self, P_a, P_b):
        """
        Compute bidirectional inclusion product between two sets of projectors.
        incl(A⊆B) * incl(B⊆A) for each (a, b) pair.

        Args:
            P_a: (N, D, D)
            P_b: (M, D, D)
        Returns:
            scores: (N, M)
        """
        N, D, _ = P_a.shape
        M = P_b.shape[0]
        a_flat = P_a.reshape(N, D * D)
        b_flat = P_b.reshape(M, D * D)
        overlap = a_flat @ b_flat.t()  # (N, M) — Tr(P_a @ P_b)
        tr_a = torch.einsum("nii->n", P_a).clamp(min=1e-6)  # (N,)
        tr_b = torch.einsum("mii->m", P_b).clamp(min=1e-6)  # (M,)
        incl_a_in_b = overlap / tr_a.unsqueeze(1)  # (N, M)
        incl_b_in_a = overlap / tr_b.unsqueeze(0)  # (N, M)
        return incl_a_in_b * incl_b_in_a  # (N, M)

    # ------------------------------------------------------------------ #
    #  Resolution methods
    # ------------------------------------------------------------------ #
    def _resolve_attr(self, raw_X_attr, raw_P_attr, singleton=False):
        """
        Resolve attr proposal against memory candidates.

        Supports two selection methods (configured via self.resolution_method):
          "argmax"         — hard argmax + STE (default)
          "gumbel_softmax" — Gumbel-Softmax weighted combination of candidates

        Args:
            raw_X_attr: (B, n_attr, ambient_dim) — proposal basis vectors
            raw_P_attr: (B, D, D) — proposal projector
            singleton:  if True, use reduced candidates (no zeros)

        Returns:
            resolved_X_attr: (B, n_attr, ambient_dim) — resolved basis
            resolved_P_attr: (B, D, D) — projector of resolved basis
            attr_hard_idx:   (B,) — index of selected candidate (argmax of scores/weights)
            commitment_loss: scalar
            scores:          (B, C) — raw bidirectional inclusion scores
        """
        B = raw_X_attr.shape[0]

        if singleton:
            cand_X = self._get_singleton_attr_candidate_vectors()
            n_cand = self.n_singleton_attr_candidates
        else:
            cand_X = self._get_attr_candidate_vectors()
            n_cand = self.n_attr_candidates

        # Build candidate projectors
        cand_P = ridge_projector(cand_X, lbd=self.lbd)  # (C, D, D)

        # Bidirectional inclusion: incl(prop⊆cand) * incl(cand⊆prop)
        scores = self._bidirectional_inclusion(raw_P_attr, cand_P)  # (B, C)

        if self.resolution_method == "gumbel_softmax":
            # Gumbel-Softmax: differentiable soft selection
            # Use (scores + eps).log() instead of scores.log().clamp() to avoid
            # gradient explosion: d/dx log(x) = 1/x is huge when x ≈ 0, and
            # clamp doesn't fix the backward gradient in the un-clamped region.
            logits = (scores + 1e-8).log()
            weights = F.gumbel_softmax(
                logits,
                tau=self.resolution_tau, hard=True,
            )  # (B, C) — one-hot in forward, soft in backward
            # Weighted combination of candidate basis vectors
            # weights: (B, C), cand_X: (C, n_attr, D)
            resolved_X_attr = torch.einsum("bc,cnd->bnd", weights, cand_X)
            attr_hard_idx = weights.argmax(dim=-1)  # (B,)
        else:
            # Argmax + STE (original behavior)
            with torch.no_grad():
                attr_hard_idx = scores.argmax(dim=-1)  # (B,)
            hard_X = cand_X[attr_hard_idx]  # (B, n_attr, ambient_dim)
            resolved_X_attr = raw_X_attr + (hard_X - raw_X_attr).detach()

        resolved_P_attr = ridge_projector(resolved_X_attr, lbd=self.lbd)

        # Commitment loss (VQ-VAE style, both directions) on unit-norm rows.
        # Proposals are already unit-norm; normalize memory candidates to match.
        hard_X = cand_X[attr_hard_idx]
        hard_X_norm = F.normalize(hard_X, p=2, dim=-1)
        codebook_loss = F.mse_loss(raw_X_attr.detach(), hard_X_norm)
        encoder_loss = F.mse_loss(raw_X_attr, hard_X_norm.detach())
        commitment_loss = (self.commitment_codebook_weight * codebook_loss
                           + self.commitment_encoder_weight * encoder_loss)

        return resolved_X_attr, resolved_P_attr, attr_hard_idx, commitment_loss, scores

    def _resolve_obj(self, raw_X_obj, raw_P_obj, singleton=False):
        """
        Resolve obj proposal against memory candidates.

        Supports two selection methods (configured via self.resolution_method):
          "argmax"         — hard argmax + STE (default)
          "gumbel_softmax" — Gumbel-Softmax weighted combination of candidates

        Args:
            raw_X_obj: (B, n_obj, ambient_dim) — proposal basis vectors
            raw_P_obj: (B, D, D) — proposal projector
            singleton: if True, use base subspaces only (no unions)

        Returns:
            resolved_X_obj: (B, n_obj, ambient_dim) — resolved basis
            resolved_P_obj: (B, D, D) — resolved projector
            obj_hard_idx:   (B,) — index of selected candidate (argmax of scores/weights)
            commitment_loss: scalar
            scores:          (B, C) — raw bidirectional inclusion scores
        """
        B = raw_X_obj.shape[0]

        if singleton:
            cand_P = self._get_singleton_obj_candidate_projectors()
            n_cand = self.n_singleton_obj_candidates
            candidate_subsets = self._singleton_obj_candidate_subsets
        else:
            cand_P = self._get_obj_candidate_projectors()
            n_cand = self.n_obj_candidates
            candidate_subsets = self._obj_candidate_subsets

        # Bidirectional inclusion: incl(prop⊆cand) * incl(cand⊆prop)
        scores = self._bidirectional_inclusion(raw_P_obj, cand_P)  # (B, C)

        if self.resolution_method == "gumbel_softmax":
            # Gumbel-Softmax: differentiable soft selection
            # Use (scores + eps).log() to avoid gradient explosion at near-zero scores.
            logits = (scores + 1e-8).log()
            weights = F.gumbel_softmax(
                logits,
                tau=self.resolution_tau, hard=True,
            )  # (B, C) — one-hot in forward, soft in backward
            obj_hard_idx = weights.argmax(dim=-1)  # (B,)
            # Weighted combination of candidate projectors
            # weights: (B, C), cand_P: (C, D, D)
            resolved_P_obj = torch.einsum("bc,cde->bde", weights, cand_P)
        else:
            # Argmax + STE (original behavior)
            with torch.no_grad():
                obj_hard_idx = scores.argmax(dim=-1)  # (B,)
            hard_P = cand_P[obj_hard_idx]  # (B, D, D)
            resolved_P_obj = raw_P_obj + (hard_P - raw_P_obj).detach()

        # Resolved X_obj: from base memory subspaces for downstream use
        base_X = self.obj_memory  # (n_obj, n_obj, ambient_dim)
        if singleton:
            resolved_X_obj = base_X[obj_hard_idx]  # (B, n_obj, ambient_dim)
        else:
            resolved_X_list = []
            for b in range(B):
                subset = candidate_subsets[obj_hard_idx[b].item()]
                parts = [base_X[i] for i in subset]
                resolved_X_list.append(torch.stack(parts).mean(dim=0))
            resolved_X_obj = torch.stack(resolved_X_list)  # (B, n_obj, ambient_dim)

        # Commitment loss (VQ-VAE style, both directions) on unit-norm rows.
        # Proposals are already unit-norm; normalize memory candidates to match.
        hard_X = resolved_X_obj  # (B, n_obj, ambient_dim) — from memory
        hard_X_norm = F.normalize(hard_X, p=2, dim=-1)
        codebook_loss = F.mse_loss(raw_X_obj.detach(), hard_X_norm)
        encoder_loss = F.mse_loss(raw_X_obj, hard_X_norm.detach())
        commitment_loss = (self.commitment_codebook_weight * codebook_loss
                           + self.commitment_encoder_weight * encoder_loss)

        return resolved_X_obj, resolved_P_obj, obj_hard_idx, commitment_loss, scores

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, R_subset, tau=1.0, hard=False, singleton=False):
        """
        Args:
            R_subset: (B, S, P, E) — perceptual representations of a subset
                      of S images, each with P patches of dim E.
            tau:      unused (kept for API compatibility)
            hard:     unused (always hard argmax + STE)
            singleton: if True, use reduced candidate sets (no zeros for attr,
                       no unions for obj).

        Returns: dict with keys:
            resolved_X_attr:    (B, n_attr, ambient_dim) — resolved attr basis
            resolved_X_obj:     (B, n_obj, ambient_dim) — resolved obj basis
            resolved_P_attr:    (B, D, D) — resolved attr projector
            resolved_P_obj:     (B, D, D) — resolved obj projector
            proposal_X_attr:    (B, n_attr, ambient_dim) — raw proposal attr basis
            proposal_X_obj:     (B, n_obj, ambient_dim) — raw proposal obj basis
            proposal_P_attr:    (B, D, D) — raw proposal attr projector
            proposal_P_obj:     (B, D, D) — raw proposal obj projector
            attr_assign_idx:    (B,) — index into attr candidates
            obj_assign_idx:     (B,) — index into obj candidates
            commitment_loss:    scalar
            proposal_norm_loss: scalar — L2 regularization on raw proposals
        """
        B, S, P, E = R_subset.shape

        # ─ 1. Flatten subset into token sequence ────────────────────────
        H = R_subset.view(B, S * P, E)  # (B, S*P, E)

        # ─ 2. Proposal: attend to patches, project to ambient space ─────
        q_attr = self.query_attr.unsqueeze(0).expand(B, -1, -1)  # (B, n_attr, E)
        q_obj = self.query_obj.unsqueeze(0).expand(B, -1, -1)    # (B, n_obj, E)

        if not self.separate_obj_attr_attn:
            # Level 0: shared decoder, concatenate attr+obj queries
            q_all = torch.cat([q_attr, q_obj], dim=1)  # (B, n_attr+n_obj, E)
            out = self.shared_decoder_norm(
                self.shared_decoder(tgt=q_all, memory=H)
            )
            x_attr_embed = out[:, :self.n_attr]  # (B, n_attr, E)
            x_obj_embed = out[:, self.n_attr:]   # (B, n_obj, E)
            x_attr_embeds = [x_attr_embed[:, i] for i in range(self.n_attr)]
        elif not self.separate_slot_attn:
            # Level 1: separate decoders for attr vs obj
            x_attr_embed = self.attr_decoder_norm(
                self.attr_decoder(tgt=q_attr, memory=H)
            )
            x_obj_embed = self.obj_decoder_norm(
                self.obj_decoder(tgt=q_obj, memory=H)
            )
            x_attr_embeds = [x_attr_embed[:, i] for i in range(self.n_attr)]
        else:
            # Level 2: per-slot decoders for attr + separate obj decoder
            x_attr_embeds = []
            for slot_idx in range(self.n_attr):
                q_slot = q_attr[:, slot_idx:slot_idx+1]  # (B, 1, E)
                x_slot = self.slot_decoder_norms[slot_idx](
                    self.slot_decoders[slot_idx](tgt=q_slot, memory=H)
                )  # (B, 1, E)
                x_attr_embeds.append(x_slot.squeeze(1))  # (B, E)
            x_obj_embed = self.obj_decoder_norm(
                self.obj_decoder(tgt=q_obj, memory=H)
            )

        raw_X_attr_parts = []
        for slot_idx in range(self.n_attr):
            d = self.dims_per_slot[slot_idx]
            offset_before = sum(self.dims_per_slot[:slot_idx])
            offset_after = self.ambient_dim - offset_before - d
            slot_proposal = self.slot_projectors[slot_idx](
                x_attr_embeds[slot_idx]
            )
            raw_X_attr_parts.append(
                F.pad(slot_proposal, (offset_before, offset_after))
            )
        raw_X_attr = torch.stack(raw_X_attr_parts, dim=1)  # (B, n_attr, ambient_dim)
        raw_X_attr = F.normalize(raw_X_attr, p=2, dim=-1)  # unit-norm rows
        raw_P_attr = ridge_projector(raw_X_attr, lbd=self.lbd)

        raw_X_obj = self.obj_projector(x_obj_embed)  # (B, n_obj, ambient_dim)
        raw_X_obj = F.normalize(raw_X_obj, p=2, dim=-1)  # unit-norm rows
        raw_P_obj = ridge_projector(raw_X_obj, lbd=self.lbd)

        # ─ 3. Resolve against memory ────────────────────────────────────
        resolved_X_attr, resolved_P_attr, attr_hard_idx, attr_commit, attr_scores = \
            self._resolve_attr(raw_X_attr, raw_P_attr, singleton=singleton)

        resolved_X_obj, resolved_P_obj, obj_hard_idx, obj_commit, obj_scores = \
            self._resolve_obj(raw_X_obj, raw_P_obj, singleton=singleton)

        commitment_loss = (attr_commit + obj_commit) / 2.0

        # Proposal norm penalty: mean squared row-norm of raw proposals.
        # Regularizes proposal scale to prevent unconstrained growth.
        attr_row_norms = raw_X_attr.norm(p=2, dim=-1)  # (B, n_attr)
        obj_row_norms = raw_X_obj.norm(p=2, dim=-1)    # (B, n_obj)
        proposal_norm_loss = (attr_row_norms.pow(2).mean() + obj_row_norms.pow(2).mean()) / 2.0

        return {
            "resolved_X_attr": resolved_X_attr,
            "resolved_X_obj": resolved_X_obj,
            "resolved_P_attr": resolved_P_attr,
            "resolved_P_obj": resolved_P_obj,
            "proposal_X_attr": raw_X_attr,
            "proposal_X_obj": raw_X_obj,
            "proposal_P_attr": raw_P_attr,
            "proposal_P_obj": raw_P_obj,
            "attr_assign_idx": attr_hard_idx,
            "obj_assign_idx": obj_hard_idx,
            "commitment_loss": commitment_loss,
            "proposal_norm_loss": proposal_norm_loss,
            "attr_scores": attr_scores,
            "obj_scores": obj_scores,
        }

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def get_attr_candidate_tuples(self):
        """Return the list of (slot_0_choice, slot_1_choice, ...) tuples.
        Choice 0 = zero, 1..K = memory fillers."""
        return self._attr_candidate_tuples

    def get_obj_candidate_subsets(self):
        """Return the list of object index subsets."""
        return self._obj_candidate_subsets

    def get_slot_memory_padded(self, slot_idx):
        """Return memory entries for one slot, L2-normalized and padded to ambient_dim."""
        memory = self.attr_memory[slot_idx]
        memory = F.normalize(memory, p=2, dim=-1)
        d = self.dims_per_slot[slot_idx]
        offset_before = sum(self.dims_per_slot[:slot_idx])
        offset_after = self.ambient_dim - offset_before - d
        return F.pad(memory, (offset_before, offset_after))

        