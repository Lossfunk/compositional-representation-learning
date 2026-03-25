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
        # Attention pools the variable-length sequence H into fixed n_attr vectors
        x_attr, _ = self.attn(query=q_attr, key=H, value=H) # (B, n_attr, embed_dim)
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
        x_obj, _ = self.attn(query=q_obj, key=H, value=H) # (B, n_obj, embed_dim)
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
        self.resolution_method = config.get("resolution_method", "cosine")

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
        # n_obj vectors in full ambient_dim, each representing one object.
        self.obj_memory = nn.Parameter(torch.randn(self.n_obj, self.ambient_dim) * 0.1)

        # ── Proposal Network ───────────────────────────────────────────
        # Shared attention pooling (same as ConceptEncoder)
        self.query_attr = nn.Parameter(torch.randn(self.n_attr, self.embed_dim))
        self.query_obj = nn.Parameter(torch.randn(self.n_obj, self.embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.heads, batch_first=True
        )

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
        """
        import itertools as _it
        options_per_slot = [range(k + 1) for k in self.fillers_per_slot]
        self._attr_candidate_tuples = list(_it.product(*options_per_slot))
        self.n_attr_candidates = len(self._attr_candidate_tuples)

    def _build_obj_candidates_index(self):
        """
        Build all non-empty subsets of object memory indices.
        For n_obj objects: 2^n_obj - 1 candidates.
        Each candidate is a tuple of object indices to union.
        """
        self._obj_candidate_subsets = []
        for r in range(1, self.n_obj + 1):
            import itertools as _it
            for subset in _it.combinations(range(self.n_obj), r):
                self._obj_candidate_subsets.append(subset)
        self.n_obj_candidates = len(self._obj_candidate_subsets)

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

        self._singleton_obj_candidate_subsets = [(i,) for i in range(self.n_obj)]
        self.n_singleton_obj_candidates = self.n_obj

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
                    # Memory filler (1-indexed → 0-indexed)
                    slot_vec = self.attr_memory[slot_idx][choice - 1]  # (d,)
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
                # choice is 1-indexed (1..K) — no zero option
                slot_vec = self.attr_memory[slot_idx][choice - 1]  # (d,)
                vec = F.pad(slot_vec, (offset_before, offset_after))
                basis_vecs.append(vec)
            candidates.append(torch.stack(basis_vecs))  # (n_attr, ambient_dim)
        return torch.stack(candidates)  # (n_singleton_attr_candidates, n_attr, ambient_dim)

    def _get_singleton_obj_candidate_projectors(self):
        """
        Build projectors for individual object memory entries (no unions).
        Returns:
            candidates_P: (n_obj, ambient_dim, ambient_dim)
            candidates_X: (n_obj, n_obj, ambient_dim) — padded
        """
        candidates_P = []
        candidates_X = []
        for subset in self._singleton_obj_candidate_subsets:
            X = self.obj_memory[list(subset)]  # (1, ambient_dim)
            P = ridge_projector(X.unsqueeze(0), lbd=self.lbd).squeeze(0)
            candidates_P.append(P)
            padded = torch.zeros(self.n_obj, self.ambient_dim, device=self.obj_memory.device)
            padded[:len(subset)] = X
            candidates_X.append(padded)
        return torch.stack(candidates_P), torch.stack(candidates_X)

    def _get_obj_candidate_projectors(self):
        """
        Build (n_obj_candidates, ambient_dim, ambient_dim) tensor of union
        projectors for each non-empty subset of object memory.
        Also returns the basis vectors for each candidate.
        """
        candidates_P = []
        candidates_X = []
        for subset in self._obj_candidate_subsets:
            X = self.obj_memory[list(subset)]  # (|subset|, ambient_dim)
            P = ridge_projector(X.unsqueeze(0), lbd=self.lbd).squeeze(0)
            candidates_P.append(P)
            # Pad X to n_obj dims for consistent shape
            padded = torch.zeros(self.n_obj, self.ambient_dim, device=self.obj_memory.device)
            padded[:len(subset)] = X
            candidates_X.append(padded)
        return torch.stack(candidates_P), torch.stack(candidates_X)

    # ------------------------------------------------------------------ #
    #  Distance / similarity scoring
    # ------------------------------------------------------------------ #
    def _cosine_score_attr(self, proposal_X_attr, candidate_X_attr):
        """
        Score each candidate by average cosine similarity across attr slots.

        Args:
            proposal_X_attr:   (B, n_attr, ambient_dim)
            candidate_X_attr:  (C, n_attr, ambient_dim)
        Returns:
            scores: (B, C)
        """
        # Flatten to (B, n_attr*ambient_dim) and (C, n_attr*ambient_dim)
        B = proposal_X_attr.shape[0]
        C = candidate_X_attr.shape[0]
        p_flat = proposal_X_attr.reshape(B, -1)
        c_flat = candidate_X_attr.reshape(C, -1)
        p_norm = F.normalize(p_flat, dim=-1)
        c_norm = F.normalize(c_flat, dim=-1)
        return p_norm @ c_norm.t()  # (B, C)

    def _inclusion_score_attr(self, proposal_P_attr, candidate_X_attr):
        """
        For each candidate, compute inclusion of candidate's attr subspace
        within the proposal's attr subspace.

        Args:
            proposal_P_attr:  (B, ambient_dim, ambient_dim) — proposal projector
            candidate_X_attr: (C, n_attr, ambient_dim) — candidate basis vectors
        Returns:
            scores: (B, C)
        """
        C = candidate_X_attr.shape[0]
        # Build candidate projectors
        cand_P = ridge_projector(candidate_X_attr, lbd=self.lbd)  # (C, D, D)

        # Inclusion(cand ⊆ proposal) = Tr(P_cand @ P_proposal) / Tr(P_cand)
        # Vectorize: P_flat (B, D*D) @ cand_P_flat.T (D*D, C)
        B, D, _ = proposal_P_attr.shape
        p_flat = proposal_P_attr.reshape(B, D * D)
        c_flat = cand_P.reshape(C, D * D)
        overlap = p_flat @ c_flat.t()  # (B, C)
        tr_cand = torch.einsum("cii->c", cand_P).clamp(min=1e-6)  # (C,)
        return overlap / tr_cand.unsqueeze(0)  # (B, C)

    def _inclusion_score_obj(self, proposal_P_obj, candidate_P_obj):
        """
        For each obj candidate (union projector), compute inclusion of
        proposal's obj subspace within the candidate's obj subspace.

        Args:
            proposal_P_obj:  (B, D, D) — proposal projector
            candidate_P_obj: (C, D, D) — candidate union projectors
        Returns:
            scores: (B, C)
        """
        B, D, _ = proposal_P_obj.shape
        C = candidate_P_obj.shape[0]
        # Inclusion(proposal ⊆ candidate) = Tr(P_proposal @ P_candidate) / Tr(P_proposal)
        p_flat = proposal_P_obj.reshape(B, D * D)
        c_flat = candidate_P_obj.reshape(C, D * D)
        overlap = p_flat @ c_flat.t()  # (B, C)
        tr_prop = torch.einsum("bii->b", proposal_P_obj).clamp(min=1e-6)  # (B,)
        return overlap / tr_prop.unsqueeze(1)  # (B, C)

    def _cosine_score_obj(self, proposal_X_obj, candidate_X_obj):
        """
        Score each obj candidate by cosine similarity.

        Args:
            proposal_X_obj:  (B, n_obj, ambient_dim)
            candidate_X_obj: (C, n_obj, ambient_dim)
        Returns:
            scores: (B, C)
        """
        B = proposal_X_obj.shape[0]
        C = candidate_X_obj.shape[0]
        p_flat = proposal_X_obj.reshape(B, -1)
        c_flat = candidate_X_obj.reshape(C, -1)
        p_norm = F.normalize(p_flat, dim=-1)
        c_norm = F.normalize(c_flat, dim=-1)
        return p_norm @ c_norm.t()  # (B, C)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, R_subset, tau=1.0, hard=False, singleton=False):
        """
        Args:
            R_subset: (B, S, P, E) — perceptual representations of a subset
                      of S images, each with P patches of dim E.
            tau:  Gumbel-Softmax temperature.
            hard: hard assignment (straight-through).
            singleton: if True, use reduced candidate sets (no zeros for attr,
                       no unions for obj). Should be used for singleton concepts.

        Returns:
            X_attr:           (B, n_attr, ambient_dim) — resolved attr basis
            X_obj:            (B, n_obj, ambient_dim) — resolved obj basis
            P_attr:           (B, D, D) — resolved attr projector
            P_obj:            (B, D, D) — resolved obj projector
            attr_assign_idx:  (B,) — index into attr candidates
            obj_assign_idx:   (B,) — index into obj candidates
            attr_log_probs:   (B,) — log-prob of chosen attr candidate
            obj_log_probs:    (B,) — log-prob of chosen obj candidate
            commitment_loss:  scalar
        """
        B, S, P, E = R_subset.shape

        # ─ 1. Flatten subset into token sequence ────────────────────────
        H = R_subset.view(B, S * P, E)  # (B, S*P, E)

        # ─ 2. Proposal: attend to patches, project to ambient space ─────
        q_attr = self.query_attr.unsqueeze(0).expand(B, -1, -1)  # (B, n_attr, E)
        x_attr_embed, _ = self.attn(query=q_attr, key=H, value=H)  # (B, n_attr, E)

        q_obj = self.query_obj.unsqueeze(0).expand(B, -1, -1)  # (B, n_obj, E)
        x_obj_embed, _ = self.attn(query=q_obj, key=H, value=H)  # (B, n_obj, E)

        # Project to ambient space — raw proposals (pre-resolution)
        raw_X_attr_parts = []
        for slot_idx in range(self.n_attr):
            d = self.dims_per_slot[slot_idx]
            offset_before = sum(self.dims_per_slot[:slot_idx])
            offset_after = self.ambient_dim - offset_before - d
            slot_proposal = self.slot_projectors[slot_idx](
                x_attr_embed[:, slot_idx]
            )  # (B, d)
            raw_X_attr_parts.append(
                F.pad(slot_proposal, (offset_before, offset_after))
            )  # (B, ambient_dim)
        raw_X_attr = torch.stack(raw_X_attr_parts, dim=1)  # (B, n_attr, ambient_dim)
        raw_P_attr = ridge_projector(raw_X_attr, lbd=self.lbd)  # (B, D, D)

        raw_X_obj = self.obj_projector(x_obj_embed)  # (B, n_obj, ambient_dim)
        raw_P_obj = ridge_projector(raw_X_obj, lbd=self.lbd)  # (B, D, D)

        # ─ 3. Resolve attr against memory candidates ────────────────────
        if singleton:
            cand_X_attr = self._get_singleton_attr_candidate_vectors()
            n_cand_attr = self.n_singleton_attr_candidates
        else:
            cand_X_attr = self._get_attr_candidate_vectors()
            n_cand_attr = self.n_attr_candidates

        if self.resolution_method == "cosine":
            attr_logits = self._cosine_score_attr(raw_X_attr, cand_X_attr)
        else:
            attr_logits = self._inclusion_score_attr(raw_P_attr, cand_X_attr)

        attr_soft = F.gumbel_softmax(attr_logits, tau=tau, hard=hard, dim=-1)  # (B, C_attr)
        # Resolved X_attr = weighted combination of candidates
        cand_X_attr_flat = cand_X_attr.reshape(n_cand_attr, -1)
        resolved_X_attr = (attr_soft @ cand_X_attr_flat).reshape(
            B, self.n_attr, self.ambient_dim
        )
        resolved_P_attr = ridge_projector(resolved_X_attr, lbd=self.lbd)

        attr_hard_idx = attr_logits.argmax(dim=-1)  # (B,)
        attr_log_probs = F.log_softmax(attr_logits, dim=-1).gather(
            1, attr_hard_idx.unsqueeze(1)
        ).squeeze(1)

        # ─ 4. Resolve obj against memory candidates ─────────────────────
        if singleton:
            cand_P_obj, cand_X_obj = self._get_singleton_obj_candidate_projectors()
            n_cand_obj = self.n_singleton_obj_candidates
        else:
            cand_P_obj, cand_X_obj = self._get_obj_candidate_projectors()
            n_cand_obj = self.n_obj_candidates

        if self.resolution_method == "cosine":
            obj_logits = self._cosine_score_obj(raw_X_obj, cand_X_obj)
        else:
            obj_logits = self._inclusion_score_obj(raw_P_obj, cand_P_obj)

        obj_soft = F.gumbel_softmax(obj_logits, tau=tau, hard=hard, dim=-1)  # (B, C_obj)
        # Resolved X_obj = weighted combination of candidate basis sets
        cand_X_obj_flat = cand_X_obj.reshape(n_cand_obj, -1)
        resolved_X_obj = (obj_soft @ cand_X_obj_flat).reshape(
            B, self.n_obj, self.ambient_dim
        )
        resolved_P_obj = ridge_projector(resolved_X_obj, lbd=self.lbd)

        obj_hard_idx = obj_logits.argmax(dim=-1)  # (B,)
        obj_log_probs = F.log_softmax(obj_logits, dim=-1).gather(
            1, obj_hard_idx.unsqueeze(1)
        ).squeeze(1)

        # ─ 5. Commitment loss ───────────────────────────────────────────
        attr_target = F.one_hot(attr_hard_idx, n_cand_attr).float()
        obj_target = F.one_hot(obj_hard_idx, n_cand_obj).float()
        commitment_loss = (
            F.mse_loss(F.softmax(attr_logits, dim=-1), attr_target.detach())
            + F.mse_loss(F.softmax(obj_logits, dim=-1), obj_target.detach())
        ) / 2.0

        return (
            resolved_X_attr, resolved_X_obj,
            resolved_P_attr, resolved_P_obj,
            attr_hard_idx, obj_hard_idx,
            attr_log_probs, obj_log_probs,
            commitment_loss,
        )

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
        """Return memory entries for one slot, padded to ambient_dim."""
        memory = self.attr_memory[slot_idx]
        d = self.dims_per_slot[slot_idx]
        offset_before = sum(self.dims_per_slot[:slot_idx])
        offset_after = self.ambient_dim - offset_before - d
        return F.pad(memory, (offset_before, offset_after))

        