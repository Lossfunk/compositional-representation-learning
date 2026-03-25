# Experiment Set 002: Breaking Binary Uniformity

## Motivation

Experiments 70-75 revealed the **core failure mode** of the current SCL architecture: at low Gumbel temperature (τ→0.1), the MLP attr encoder produces the **same binary code** for all concepts regardless of cardinality. This causes:

1. **Attr rank saturation**: All cardinalities collapse to ~1.905 (= 2 × 20/21), with no differentiation between singletons and combined concepts
2. **Attr heatmap uniformity**: All inclusion values become ~0.72-0.95, destroying discriminative power
3. **False IC convergence**: IC loss → 0 trivially because all codes are identical, not because structure is correct
4. **Training crashes**: Gradient instability from compounding structural losses during gumbel annealing causes linalg.solve failures

Meanwhile, the **object pathway succeeds** (exp_72 achieved obj_rank_8=3.975 ≈ ideal 4.0) because it has no gumbel/mask bottleneck. This proves the architecture CAN learn correct lattice structure — the attr pathway is specifically bottlenecked.

Additionally, the **ridge regularization floor** at λ=0.05 creates a minimum non-zero rank of 20/21 ≈ 0.952 per active dimension, preventing attr ranks from reaching their ideal values.

## Three Interventions

### A. Lower ridge λ (reduce rank floor)
With λ=0.05, rank per active dim = 1/1.05 = 0.952. Lowering λ:
- λ=0.01 → rank = 1/1.01 = 0.990 (floor of 0.990 vs 0.952)
- λ=0.001 → rank = 1/1.001 = 0.999

This makes rank values more precise and allows a wider effective range, especially for combined concepts that should have near-zero attr rank.

### B. Disable reconstruction (free attr from visual encoding pressure)
The reconstruction loss dominates early training (~2000 → ~20), creating a gradient basin that shapes attr_enc toward visual encoding. This is counterproductive for structural differentiation: reconstruction wants rich, continuous representations while structure wants sparse, discriminative binary codes.

Without reconstruction, the attr pathway is freed to focus on lattice structure via galois, IC, sink, and rank ordering losses. The Galois connection to the obj space (which IS grounded visually) should prevent the attr codes from becoming arbitrary.

### C. Softer gumbel annealing (end temp 0.3 instead of 0.1)
At τ=0.1, the sigmoid is nearly a step function — any positive input → 1. At τ=0.3, outputs are softer (~0.73 vs ~0.27), giving the model room to differentiate:
- Singleton with strong signal → ~0.73
- Combined concept with averaged signal → ~0.50
- This creates measurably different projector ranks

## Baseline Experiments

### exp_72 (primary baseline)
Best obj ranks ever: obj_rank_8=3.975. Clean config: MLP encoder, rank-only basis_ortho=1, basis_sparsity=1, IC=10, repulsion_obj=1. Crashed at step 967 but ran long enough to establish strong trends.

### exp_73 (secondary baseline)
Only experiment showing attr rank differentiation (card_8=0.976 vs card_1=1.696) due to rank_conservation=1. Higher reconstruction (77.2) suggests the rank_conservation fights visual encoding. Same config as exp_72 + rank_conservation_loss=1.

## Experiment Design

### Group A: Lower ridge λ

Tests whether reducing the ridge regularization improves rank precision. Built on exp_72.

| ID | Name | λ | Base | Other changes | Expected impact |
|----|------|---|------|---------------|----------------|
| 76 | λ=0.01 | **0.01** | exp_72 | — | Singleton rank: 1.980 (vs 1.905). Floor per dim: 0.990 (vs 0.952). More headroom for differentiation. Numerically still safe. |
| 77 | λ=0.001 | **0.001** | exp_72 | — | Singleton rank: 1.998 (vs 1.905). Floor per dim: 0.999. Maximum precision but may cause numerical instability in ridge_projector (weaker regularization). |

### Group B: Disable reconstruction

Tests whether removing reconstruction pressure allows structural losses to discover correct attr codes.

| ID | Name | recon_wt | Base | Other changes | Expected impact |
|----|------|----------|------|---------------|----------------|
| 78 | no-recon (exp_72 base) | **0.0** | exp_72 | — | Attr codes freed from visual encoding pressure. Structural losses can shape codes without fighting reconstruction. Risk: codes may become arbitrary if Galois connection doesn't provide enough grounding. |
| 79 | no-recon (exp_73 base) | **0.0** | exp_73 | — | exp_73 already showed best attr differentiation. Removing reconstruction may let rank_conservation fully separate codes. Highest potential for clean attr lattice. |

### Group C: Softer gumbel annealing

Tests whether higher end temperature preserves code differentiation by preventing the hard binary collapse.

| ID | Name | end_temp | Base | Other changes | Expected impact |
|----|------|----------|------|---------------|----------------|
| 80 | τ_end=0.3 (exp_72 base) | **0.3** | exp_72 | — | Softer binary allows intermediate values. Combined concepts can produce ~0.5 where singletons produce ~0.73. Should break the uniform-code problem. |
| 81 | τ_end=0.3 (exp_73 base) | **0.3** | exp_73 | — | exp_73's rank_conservation + softer binary. The rank_conservation loss should be more effective when codes aren't forced to be identical binary patterns. |

## Shared Config (exp_72 base)

```yaml
model:
  type: SubspaceConceptLattice
  config:
    embed_dim: 128
    ambient_dim: 4
    n_obj: 4
    n_attr: 2
    image_size: 64
    image_channels: 3
    lbd: 0.05  # varies in Group A
    max_combinations_per_cardinality: 10
    intersection_power_steps: 5
    intersection_consistency_only_pairs: False
    attr_orthogonality_avg_first: False
    attr_force_orthogonal_basis: True
    global_galois_loss_start_epoch: 0
    subspace_gumbel_sigmoid: True
    gumbel_sigmoid_annealing: True
    gumbel_sigmoid_annealing_start_epoch: 5
    gumbel_sigmoid_annealing_end_epoch: 80
    gumbel_sigmoid_annealing_start_temp: 1.0
    gumbel_sigmoid_annealing_end_temp: 0.1  # varies in Group C
    gumbel_sigmoid_hard: False
    binary_intersection: True
    basis_orthogonality_loss_type: ["rank"]
    # MLP encoder, ViT perceptual encoder/decoder (same as exp_72)
    loss_weights:
      reconstruction_loss: 1.0  # varies in Group B
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
      basis_sparsity_loss: 1.0
      loss_obj_card_prop: 1.0
      loss_attr_card_inv_prop: 1.0
      loss_attr_obj_inv_prop: 1.0
      repulsion_loss_obj: 1.0
      repulsion_loss_attr: 0.0
```

exp_73 base adds: `residual_orthogonality_loss: 0.0`, `rank_conservation_loss: 1.0`

## Key Metrics to Track

| Metric | What it tells us | Target |
|--------|-----------------|--------|
| attr rank card_1 | Singleton code quality | Near 2.0 |
| attr rank card_8 | Combined concept differentiation | Near 0.0 |
| attr rank spread (card_1 - card_8) | Differentiation success | >1.0 (ideally ~2.0) |
| obj rank card_8 | Object structure (should stay good) | Near 4.0 |
| galois_attr_loss | Galois compliance | <0.1 |
| intersection_consistency_loss | IC convergence (watch for false positives) | <0.01 |
| basis_sparsity_loss | One-hot convergence | <0.1 |
| reconstruction_loss | Visual quality (N/A for Group B) | <30 |
| attr heatmap pattern | Discriminative inclusion values | Clear 0/1 differentiation |
| basis vector plots | Code diversity across concepts | Different codes per singleton |

## Expected Outcomes

1. **Group A (λ reduction)**: Alone, this won't fix binary uniformity — it only improves rank precision. But it will be valuable COMBINED with fixes that produce different codes. λ=0.01 should be safe; λ=0.001 may cause instability.

2. **Group B (no reconstruction)**: Highest-risk, highest-reward intervention. If the Galois connection provides enough grounding, we should see the first ever experiment with genuinely different attr codes for different concepts. If grounding is insufficient, codes may become arbitrary nonsense that satisfies structural losses without semantic meaning.

3. **Group C (softer τ)**: Most conservative intervention. Should partially alleviate binary uniformity by allowing intermediate values. May not fully solve the problem (soft values can still converge to similar patterns) but should interact well with other losses.

4. **Cross-group insights**: If exp_78 or 79 (no recon) shows structural success but arbitrary codes, and exp_80 or 81 (softer τ) shows partial differentiation, the next experiment set should combine: no recon + softer τ + lower λ.

## Quick Reference Table

| Exp | Group | Base | λ | recon_wt | τ_end | rank_cons | Key question |
|-----|-------|------|---|----------|-------|-----------|-------------|
| 76 | A | 72 | **0.01** | 1.0 | 0.1 | 0 | Does lower λ improve rank precision? |
| 77 | A | 72 | **0.001** | 1.0 | 0.1 | 0 | How aggressive can we go with λ? |
| 78 | B | 72 | 0.05 | **0.0** | 0.1 | 0 | Can structural losses alone discover correct codes? |
| 79 | B | 73 | 0.05 | **0.0** | 0.1 | **1** | Best attr diff base + no recon = clean separation? |
| 80 | C | 72 | 0.05 | 1.0 | **0.3** | 0 | Does softer binary break uniformity? |
| 81 | C | 73 | 0.05 | 1.0 | **0.3** | **1** | Softer binary + rank conservation = full solution? |

## Total: 6 experiments (scl_exp_76 through scl_exp_81)
