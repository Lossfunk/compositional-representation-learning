# Experiment Set 003: Continuous Attribute Subspaces

## Motivation

Experiments 70-81 exhaustively demonstrated that the binary attr encoding pipeline (`gumbel_sigmoid` + `orthogonal_mask` + `ridge_projector`) has a **degenerate global optimum**: with `n_attr=2`, `ambient_dim=4`, all concepts converge to the same one-hot binary code `[1,1]` (both basis slots active). This produces uniform attr ranks at 1.905 (lambda floor) with zero cardinality differentiation.

**Root cause**: With only 4 possible binary codes per basis slot (`[0,0]`, `[1,0]`, `[0,1]`, `[1,1]`) and `max_singleton_attr_rank_loss` pushing toward rank=2, the model is forced to use `[1,1]` for singletons. Combined concepts also converge to `[1,1]` because no loss penalizes code uniformity, and IC/galois/sink are trivially satisfied by identical codes.

**This experiment set returns to continuous attr representations.** Without gumbel binarization, the attr encoder outputs arbitrary real-valued basis vectors. The ridge projector naturally handles continuous subspaces, and intersection/inclusion operate on projectors rather than binary codes. This gives the model continuous control over rank (via vector magnitudes and directions) rather than discrete on/off codes.

Key changes from the binary experiments:
- `subspace_gumbel_sigmoid: False` — no binarization
- `binary_intersection: False` — projector-based intersection via power iteration
- `binary_inclusion: False` — projector-based inclusion via trace ratio
- Explore `ambient_dim` from 4 to 16 (with `n_attr=2`, gives 2-8 dims per basis slot)
- `basis_sparsity_loss: 0` — targets one-hot, antithetical to continuous codes
- Re-evaluate all structural losses in the continuous setting

## Hypotheses

1. **Continuous codes break the degenerate optimum** because the loss landscape has smooth gradients for rank differentiation. Different concepts can have different vector magnitudes, producing different ranks.

2. **Larger ambient_dim helps**: With `ambient_dim=8`, each basis slot has 4 continuous dims. Singletons can use high-magnitude vectors (rank~2), while combined concepts use lower-magnitude or more aligned vectors (rank<2). The extra dimensions give room for this differentiation.

3. **Projector-based intersection is more principled** than binary min: `matrix_power(avg(P_singletons), power_steps)` converges to the true intersection projector, giving smoother gradients.

4. **Reconstruction helps rather than hurts**: In binary mode, reconstruction created a local basin. In continuous mode, reconstruction provides useful gradient signal because different visual features naturally produce different continuous codes.

## Baseline

Best binary experiment: **exp_72** (best obj ranks, attr flat at 1.905).
Best continuous reference: pre-binary experiments used continuous mode but with different loss configs.

## Experiment Design

### Group A: Ambient Dimension Sweep (core exploration)

Tests the effect of ambient_dim on continuous attr subspaces with a solid loss configuration.
All use: no gumbel, projector intersection, projector inclusion, MLP encoder.

| ID | ambient_dim | dims_per_slot | basis_ortho_rank_target | Key question |
|----|-------------|---------------|------------------------|-------------|
| 82 | 4 | 2 | 2.0 | Continuous baseline at original dim. Can continuous codes differentiate where binary couldn't? |
| 83 | 8 | 4 | 2.0 | More room per slot. Expected sweet spot. |
| 84 | 12 | 6 | 2.0 | Even more headroom. Tests if extra dims help or add noise. |
| 85 | 16 | 8 | 2.0 | Large ambient dim. Risk of underdetermination. |

### Group B: Loss Configuration Exploration (at ambient_dim=8)

Tests which structural losses matter most in continuous mode. All at ambient_dim=8.

| ID | Key changes from Group A base | Purpose |
|----|------------------------------|---------|
| 86 | `attr_sink_loss: 0` | Is attr_sink still needed? In binary mode it was crucial. |
| 87 | `intersection_consistency_loss: 0`, `union_consistency_loss: 0` | Remove lattice consistency. Do proportionality + galois alone suffice? |
| 88 | `basis_orthogonality_loss: 0` | Remove basis ortho. Does the model find orthogonal structures naturally? |
| 89 | `galois_attr_loss: 0`, `galois_obj_loss: 0` | No Galois. Do intersection+union+proportionality suffice? |
| 90 | `max_singleton_attr_rank_loss: 1.0` (reduced from 10) | Weaker singleton anchoring. Frees attr space for rank differentiation. |
| 91 | `loss_attr_card_inv_prop: 10.0` (increased from 1) | Stronger inverse proportionality pressure. Directly forces rank ordering by cardinality. |

### Group C: Intersection Method Comparison (at ambient_dim=8)

| ID | intersection_method | intersection_power_steps | Key question |
|----|-------------------|-------------------------|-------------|
| 92 | `intersection_consistency_only_pairs: True` | N/A | Pairwise projector intersection. More stable but less general. |
| 93 | Power iteration (default) | 10 (increased from 5) | More convergent intersection approximation. |

### Group D: Additional Structural Variations (at ambient_dim=8)

| ID | Key changes | Purpose |
|----|------------|---------|
| 94 | `clamp_X: True` | Restrict basis vectors to [0,1]. May help with numerical stability. |
| 95 | `repulsion_loss_obj: 0`, `repulsion_loss_attr: 1.0` | Enable attr repulsion, disable obj repulsion. Directly push attr singletons apart. |
| 96 | `rank_conservation_loss: 1.0`, `residual_orthogonality_loss: 1.0` | Add residual losses. Tests if projector-mode residual decomposition helps. |
| 97 | `attr_polarization_loss: 1.0` | Add polarization. In continuous mode, pushes basis vector similarities toward 0 or 1. |

### Group E: Combined Best (at ambient_dim=8, informed guesses)

| ID | Description | Key changes |
|----|------------|-------------|
| 98 | "Kitchen sink" | `attr_sink=10, galois=10/1, IC=10, UC=1, basis_ortho=1, card_inv=10, repulsion_attr=1, rank_cons=1` — combine strongest signals |
| 99 | "Minimal lattice" | Only `IC=10, UC=1, galois=10/1, recon=1, singleton_obj=1, max_singleton_attr=10` — can the lattice constraints alone learn structure? |

## Shared Config (Groups A-E, unless overridden)

```yaml
model:
  type: "SubspaceConceptLattice"
  config:
    embed_dim: 128
    ambient_dim: 8  # varies in Group A
    n_obj: 4
    n_attr: 2
    image_size: 64
    image_channels: 3
    lbd: 0.05
    max_combinations_per_cardinality: 10
    intersection_power_steps: 5
    intersection_consistency_only_pairs: False
    attr_orthogonality_avg_first: False
    attr_force_orthogonal_basis: True
    global_galois_loss_start_epoch: 0
    subspace_gumbel_sigmoid: False   # KEY: no binarization
    binary_intersection: False        # KEY: projector intersection
    binary_inclusion: False           # KEY: projector inclusion
    basis_orthogonality_loss_type: ["rank"]
    basis_orthogonality_rank_target: 2.0
    perceptual_encoder:
      type: "ViTEncoder"
      config:
        patch_size: 8
        depth: 4
        heads: 8
        mlp_ratio: 4.0
    concept_encoder:
      type: "ConceptEncoder"
      config:
        mapping_type: "mlp"
        heads: 8
        clamp_X: False
    decoder:
      type: "ViTDecoder"
      config:
        patch_size: 8
        depth: 4
        heads: 8
        mlp_ratio: 4.0
    loss_weights:
      reconstruction_loss: 1.0
      singleton_obj_rank_loss: 1.0
      max_singleton_attr_rank_loss: 10.0
      modular_subspace_loss: 0.0
      attr_orthogonality_loss: 0.0
      attr_polarization_loss: 0.0
      concept_similarity_loss: 0.0
      union_consistency_loss: 1.0
      intersection_consistency_loss: 10.0
      attr_sink_loss: 10.0
      galois_attr_loss: 10.0
      galois_obj_loss: 1.0
      global_galois_loss: 0.0
      basis_orthogonality_loss: 1.0
      basis_sparsity_loss: 0.0       # KEY: disabled for continuous
      loss_obj_card_prop: 1.0
      loss_attr_card_inv_prop: 1.0
      loss_attr_obj_inv_prop: 1.0
      repulsion_loss_obj: 1.0
      repulsion_loss_attr: 0.0
      residual_orthogonality_loss: 0.0
      rank_conservation_loss: 0.0

data:
  train:
    type: v0Dataset
    config:
      image_size: [64, 64]
      shapes: ["circle", "square"]
      colors: ["red", "blue"]
      excluded_combinations: []
      num_samples: 1000
      return_metadata: True
      center_range: [32, 33]
      size_range: [20, 21]
    dataloader_config:
      batch_size: 8
      shuffle: True

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
| attr rank card_1 | ~2.0 | Singleton should span full attr subspace |
| attr rank card_8 | ~0.0 | Universal concept should have near-empty attr intersection |
| attr rank spread (card_1 - card_8) | >1.0 | Must differentiate by cardinality |
| obj rank card_1 / card_8 | ~1.0 / ~4.0 | Object structure quality |
| reconstruction_loss | <1.0 | Visual grounding |
| intersection_consistency_loss | Decreasing | Lattice structure forming |
| galois_attr_loss | Decreasing | Galois connection satisfied |
| basis_orthogonality_loss | Decreasing | Between-concept orthogonality |

## Logging Note

Concept basis vector plots remain enabled — with ambient_dim up to 16 and n_attr=2, these are still small bar charts (2 x 16 = 32 values). The W&B image logging overhead is negligible. No changes needed.

## Code Changes Required

1. **`basis_orthogonality_rank_target`**: Made configurable (previously hardcoded at 2.0). Falls back to 2.0 if not specified.

## Quick Reference Table

| Exp | Group | ambient_dim | Key variation | Priority |
|-----|-------|-------------|--------------|----------|
| 82 | A | 4 | Continuous baseline | High |
| 83 | A | 8 | Expected sweet spot | **Critical** |
| 84 | A | 12 | Higher dim | Medium |
| 85 | A | 16 | Large dim | Low |
| 86 | B | 8 | No attr_sink | Medium |
| 87 | B | 8 | No IC/UC | Medium |
| 88 | B | 8 | No basis_ortho | Medium |
| 89 | B | 8 | No Galois | Medium |
| 90 | B | 8 | Weak singleton anchoring | High |
| 91 | B | 8 | Strong inv proportionality | High |
| 92 | C | 8 | Pairwise intersection | Medium |
| 93 | C | 8 | Power steps=10 | Low |
| 94 | D | 8 | clamp_X=True | Medium |
| 95 | D | 8 | Attr repulsion on, obj off | High |
| 96 | D | 8 | Residual + conservation | Medium |
| 97 | D | 8 | Polarization | Medium |
| 98 | E | 8 | Kitchen sink | High |
| 99 | E | 8 | Minimal lattice | High |

## Total: 18 experiments (scl_exp_82 through scl_exp_99)
