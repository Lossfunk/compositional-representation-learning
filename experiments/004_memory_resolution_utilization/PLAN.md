# Experiment Set 004: Memory Resolution & Utilization Ablation

## Motivation

Experiments mem_scl_exp_1 through exp_3 revealed a fundamental failure mode in the MemorySubspaceConceptLattice architecture: **all singletons collapse to the same memory candidate**, preventing any attr lattice differentiation. Deep analysis (see `analysis/comparisons/mem_exp_1_3_v3/WORKING_NOTES.md`) identified the complete causal chain:

1. **All singleton attr candidates have identical ridge rank** (1.9048). With `n_attr=2`, `dims_per_slot=[4,4]`, L2-normalized attr memory, and orthogonal slot allocation, every candidate is two orthogonal unit vectors — structurally identical from a rank perspective.

2. **STE self-reinforcing lock-in**: When all singletons happen to select the same candidate early in training, the straight-through estimator pushes all proposals toward that candidate. Proposals converge → same selection scores → permanent lock-in.

3. **Softmax-based utilization hides collapse**: The original utilization loss used `softmax(scores)` which produces near-uniform distributions even when argmax always picks the same candidate. KL divergence from uniform was trivially low (0.016) despite total collapse (true KL should be 1.386).

4. **~5 losses benefit from collapse**: Commitment loss, galois, IC, reconstruction, and repulsion all achieve lower values when all singletons pick the same candidate, creating a gradient coalition that overpowers the weak utilization signal.

5. **Encoder cannot produce lower-rank proposals for multi-image concepts**: The shared MHA pools all image tokens → both attr basis slots always active → full-rank proposal → bidirectional inclusion favors full-rank candidates → no rank differentiation across cardinalities.

This experiment set ablates two orthogonal interventions designed to break the collapse:

### Intervention 1: Gumbel-Softmax Resolution (replaces argmax + STE)
Instead of deterministic argmax with straight-through gradients, use `F.gumbel_softmax(log_scores, tau, hard=True)`:
- **Forward**: one-hot selection (same discrete behavior)
- **Backward**: soft gradients through all candidates weighted by Gumbel probabilities
- **Gumbel noise**: randomly perturbs selection, breaking deterministic lock-in
- Temperature τ controls exploration: high τ = more random, low τ = near-deterministic

### Intervention 2: Honest Utilization Loss
Two alternatives to the broken softmax utilization:

**Hard (bincount)**: Count actual hard assignments via bincount, compute KL from uniform. No gradient (the loss is a monitoring signal + indirect pressure through other losses), but gives honest collapse detection. KL = 1.386 for total collapse vs 0 for perfect uniform.

**Soft (sharpened softmax)**: `softmax(scores / τ_util)` with small τ_util (0.1 or 0.01) before computing KL. Maintains differentiability while revealing the peaked structure that plain softmax hides.

## Prior Work

- **mem_scl_exp_1** (baseline, all weights=1): Total attr collapse. All singletons → same candidate. Obj partially works (color-based grouping).
- **mem_scl_exp_3** (10x structural weights, recon=0.1): Better obj differentiation but attr still collapsed. 10x attr-side losses destabilized obj via shared MHA.

## Code Changes (already implemented)

1. **`ConceptEncoder.py`**: `_resolve_attr` and `_resolve_obj` support configurable `resolution_method` (`"argmax"` or `"gumbel_softmax"`) with `resolution_tau`.
2. **`VQSubspaceConceptLattice.py`**: Separate `utilization_loss_attr` / `utilization_loss_obj` with configurable `utilization_mode` (`"hard"` or `"soft"`) and `utilization_tau`. Config weight keys: `utilization_loss_attr`, `utilization_loss_obj` (with fallback to `utilization_loss`).
3. **`ConceptEncoder.py`**: obj_memory initialization changed to rank-1 per entry (all rows identical) with mutual orthogonality between entries.

## Experiment Design

### Ablation Matrix

Two axes: **resolution method** × **utilization mode/weight**. All other config held constant (all structural loss weights = 1.0, recon = 1.0).

| Exp | Resolution | Res. τ | Utilization | Util. τ | Util. Weight | Key Question |
|-----|-----------|--------|-------------|---------|-------------|--------------|
| **10** | argmax | — | hard | — | 1 | Baseline: honest util + STE. Does honest signal alone help? |
| **11** | argmax | — | soft | 0.1 | 1 | Sharpened soft util with argmax. Differentiable collapse detection. |
| **12** | argmax | — | soft | 0.01 | 1 | Very sharp soft util. Near-hard but with gradients. |
| **13** | gumbel | 1.0 | hard | — | 1 | High-temp gumbel. Max exploration noise — does random selection break lock-in? |
| **14** | gumbel | 0.5 | hard | — | 1 | Medium gumbel. Balanced exploration vs exploitation. |
| **15** | gumbel | 0.1 | hard | — | 1 | Low-temp gumbel. Near-deterministic but differentiable through candidates. |
| **16** | gumbel | 0.5 | soft | 0.1 | 1 | Best of both: gumbel exploration + differentiable util signal. |
| **17** | argmax | — | hard | — | 10 | High util pressure. Can 10x weight overcome the ~5 collapse-favoring losses? |
| **18** | gumbel | 0.5 | hard | — | 10 | High util pressure + gumbel. Strongest intervention combo. |

### Key Comparisons

- **10 vs 13/14/15**: Does Gumbel noise break STE lock-in? Which temperature works best?
- **10 vs 11/12**: Does differentiable utilization provide useful gradients through argmax+STE?
- **14 vs 16**: Does adding soft utilization on top of Gumbel help beyond hard monitoring?
- **10 vs 17, 14 vs 18**: Does 10x utilization weight overcome the gradient coalition favoring collapse?
- **13 vs 14 vs 15**: Gumbel temperature sensitivity — exploration vs stability tradeoff.

## Shared Config

```yaml
model:
  type: MemorySubspaceConceptLattice
  config:
    embed_dim: 128
    ambient_dim: 8
    n_attr: 2
    n_obj: 4
    image_size: 64
    image_channels: 3
    lbd: 0.05
    fillers_per_slot: [2, 2]
    max_combinations_per_cardinality: 10
    # resolution_method: varies (argmax | gumbel_softmax)
    # resolution_tau: varies (0.1 | 0.5 | 1.0)
    # utilization_mode: varies (hard | soft)
    # utilization_tau: varies (0.01 | 0.1)
    perceptual_encoder:
      type: ViTEncoder
      config:
        patch_size: 8
        depth: 4
        heads: 8
        mlp_ratio: 4.0
    concept_encoder:
      config:
        heads: 8
    decoder:
      type: ViTDecoder
      config:
        patch_size: 8
        depth: 4
        heads: 8
        mlp_ratio: 4.0
    loss_weights:
      reconstruction_loss: 1.0
      commitment_loss: 1.0
      memory_slot_orthogonality_loss: 1.0
      memory_obj_rank_loss: 1.0
      memory_obj_orthogonality_loss: 1.0
      max_singleton_attr_rank_loss: 1.0
      galois_attr_loss: 1.0
      galois_obj_loss: 1.0
      intersection_consistency_loss: 1.0
      union_consistency_loss: 1.0
      attr_sink_loss: 1.0
      loss_attr_obj_inv_prop: 1.0
      repulsion_loss_attr: 1.0
      repulsion_loss_obj: 1.0
      utilization_loss_attr: 1.0  # varies (1 or 10)
      utilization_loss_obj: 1.0   # varies (1 or 10)
      proposal_norm_loss: 0.0
    loss_targets:
      max_singleton_attr_rank_loss: proposal
      repulsion_loss_attr: proposal
      repulsion_loss_obj: proposal
      galois_attr_loss: proposal
      galois_obj_loss: proposal
      intersection_consistency_loss: proposal
      union_consistency_loss: proposal
      attr_sink_loss: proposal
      loss_attr_obj_inv_prop: proposal

data:
  train:
    type: v0Dataset
    config:
      image_size: [64, 64]
      shapes: [circle, square]
      colors: [red, blue]
      excluded_combinations: []
      num_samples: 1000
      return_metadata: true
      center_range: [32, 33]
      size_range: [20, 21]
    dataloader_config:
      batch_size: 8
      shuffle: true
      num_workers: 4

trainer:
  max_epochs: 100
  optimizer:
    type: Adam
    config:
      lr: 0.0001
```

## Key Metrics to Track

| Metric | Target | Why |
|--------|--------|-----|
| utilization_loss_attr (hard KL) | < 0.5 | Collapse broken — singletons use multiple candidates |
| utilization_loss_obj (hard KL) | < 0.5 | Obj candidates utilized |
| singleton attr assignments | All 4 candidates used | Direct collapse check |
| attr rank card_1 | ~1.9 | Singletons at ceiling (both slots active) |
| attr rank card_2+ | < 1.5 | Multi-image concepts differentiate |
| attr rank spread | > 0.5 | Cardinality ordering present |
| galois_attr_loss | Decreasing | Galois connection forming |
| intersection_consistency_loss | Decreasing | Lattice intersection structure |
| reconstruction_loss | Decreasing | Visual grounding maintained |

## Hypotheses

1. **Gumbel-Softmax will break lock-in** at medium τ (0.5): The random perturbation prevents all singletons from deterministically selecting the same candidate, giving utilization and repulsion losses a chance to create differentiation. Too-high τ (1.0) may prevent convergence; too-low τ (0.1) may not provide enough exploration.

2. **Hard utilization alone (exp_10) won't break collapse**: Without gradient signal from utilization, the 5 collapse-favoring losses still dominate. Hard util is monitoring, not intervention.

3. **Gumbel + high util weight (exp_18) is the strongest candidate**: Gumbel breaks the deterministic lock-in, and 10x utilization creates strong pressure toward uniform candidate usage.

4. **Soft utilization with argmax (exp_11/12) may partially help**: The sharpened softmax reveals the peaked score distribution, giving a gradient to spread scores. But argmax+STE still has the self-reinforcing loop.

5. **The obj pathway will be less affected**: Obj resolution has more candidates and less structural degeneracy (different entries have different ranks). Gumbel noise may still improve obj utilization.

## Total: 9 experiments (mem_scl_exp_10 through mem_scl_exp_18)
