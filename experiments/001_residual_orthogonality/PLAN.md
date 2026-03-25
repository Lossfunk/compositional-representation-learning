# Experiment Set 001: Residual Orthogonality & Rank Conservation

## Motivation

The SCL model has a fundamental challenge: it learns what concepts have **in common** (intersection/similarity) but has no explicit signal about what makes them **different**. This leads to:

1. **Attr rank flattening**: Bottom-half attr ranks flatten at ~0.95 (exp_51). The model doesn't push non-shared subspace dimensions apart.
2. **Weak shape discrimination**: Attr heatmaps show poor shape differentiation even when color is well-separated.
3. **Incomparable concept pairs** (e.g., red square + blue circle) default to universal concept rather than properly encoding rank-0 intersection.

**New losses introduced:**
- **`residual_orthogonality_loss`**: For each singleton pair, after computing intersection, the residual subspaces (what's unique to each concept) should be orthogonal. E.g., after removing "red" from red-circle and red-square, "circle" and "square" residuals should be orthogonal.
- **`rank_conservation_loss`**: rank(intersection) + rank(residual) = rank(original). Ensures no rank is lost or duplicated in decomposition.

**Hypothesis**: These losses will break the attr rank flattening by giving the model explicit gradient signal to differentiate the non-shared dimensions, particularly helping shape discrimination.

## Baseline: exp_51 (Best Binary Config)

From SYNTHESIS.md, exp_51 is the best binary-mode experiment:
- **Obj ranks**: 1.00-3.98 (99.5% utilization) — best ever
- **Obj heatmap**: Excellent (cross-color 0.02-0.28, wrong singletons ~0.002)
- **Attr ranks**: 0.95-1.92 — singleton well-anchored but bottom flattened
- **Attr heatmap**: Moderate (many entries at ~0.49, partial concept collapse)
- **Reconstruction**: 0.086 — good

Key config: binary_intersection=true, annealed Gumbel (1->0.1, epochs 5-80), linear encoder, force_orthogonal_basis=true, ambient_dim=4, attr_sink=10, galois_attr=10, max_singleton=10, basis_ortho=20, repulsion=1/1.

## Experiment Design

### Control group: Establish the new baseline

| ID | Name | Change from exp_51 | Purpose |
|----|------|-------------------|---------|
| 55 | exp_51 re-run | None (exact copy) | Reproducibility baseline for new experiment set |

### Group A: Residual orthogonality loss sweep

Tests the new residual_orthogonality_loss at different weights. rank_conservation_loss held at 0 to isolate the effect of residual orthogonality alone.

| ID | Name | residual_ortho | rank_conservation | Other changes | Expected impact |
|----|------|---------------|-------------------|---------------|----------------|
| 56 | res_ortho=1 | **1** | 0 | — | Mild pressure. May slightly improve attr differentiation without disrupting obj structure. |
| 57 | res_ortho=5 | **5** | 0 | — | Moderate pressure. Should meaningfully push residual subspaces apart. Key experiment. |
| 58 | res_ortho=10 | **10** | 0 | — | Strong pressure. Risk of competing with reconstruction. Important to find the sweet spot. |

### Group B: Rank conservation loss sweep

Tests rank_conservation_loss alone. This is the "no rank should be lost" constraint.

| ID | Name | residual_ortho | rank_conservation | Other changes | Expected impact |
|----|------|---------------|-------------------|---------------|----------------|
| 59 | rank_cons=1 | 0 | **1** | — | Mild conservation. May help attr ranks fill the full 0-2 range by preventing rank "leakage". |
| 60 | rank_cons=5 | 0 | **5** | — | Moderate conservation. Should create stronger pressure for proper rank decomposition. |

### Group C: Combined new losses

Tests both losses together at the best weights identified from Groups A and B (or reasonable estimates).

| ID | Name | residual_ortho | rank_conservation | Other changes | Expected impact |
|----|------|---------------|-------------------|---------------|----------------|
| 61 | combined_5_1 | **5** | **1** | — | Combined moderate orthogonality + mild conservation. Expected best combination. |
| 62 | combined_10_5 | **10** | **5** | — | Combined strong. Risk of over-constraining, but may produce strongest attr differentiation. |

### Group D: Singleton anchoring interaction (critical open question from synthesis)

The synthesis identified a key tradeoff: max_singleton_attr_rank=10x pushes card_1 to ~1.92 but flattens the bottom. Can the new losses fix the flattening while keeping 10x anchoring? Also test the "ideal" config: binary_intersection + 1x singleton (never tested).

| ID | Name | residual_ortho | rank_conservation | max_singleton | Other changes | Expected impact |
|----|------|---------------|-------------------|--------------|---------------|----------------|
| 63 | singleton=1x + new losses | **5** | **1** | **1** | — | **High priority**. Tests whether binary_intersection + 1x singleton + new losses can achieve both good attr bottom differentiation (exp_47's card_8~0.02) AND good obj ranks (exp_51's 99.5%). This is the "ideal" experiment from synthesis open question #1. |
| 64 | singleton=1x, no new losses | 0 | 0 | **1** | — | Control for exp_63. Binary_intersection + 1x singleton without new losses. Directly answers synthesis question #1. |
| 65 | singleton=5x + new losses | **5** | **1** | **5** | — | Intermediate singleton weight. Tests whether 5x is a better balance point than 1x or 10x. |

### Group E: Polarization interaction (high priority open question from synthesis)

attr_polarization_loss was the strongest tool for obj heatmaps in Phase 7-8 but was never tested with binary_intersection or annealed Gumbel. High priority from synthesis.

| ID | Name | residual_ortho | rank_conservation | polarization | Other changes | Expected impact |
|----|------|---------------|-------------------|-------------|---------------|----------------|
| 66 | polarization=1 | 0 | 0 | **1** | — | Control: polarization alone with binary_intersection. May dramatically sharpen both heatmaps. |
| 67 | polarization=1 + new losses | **5** | **1** | **1** | — | Combined: polarization + new losses. Could be the best overall config if polarization and residual orthogonality are complementary. |

## Shared Config (all experiments unless noted)

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
    lbd: 0.05
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
    gumbel_sigmoid_annealing_end_temp: 0.1
    gumbel_sigmoid_hard: False
    binary_intersection: True
    # ... encoder, decoder configs same as exp_51
    loss_weights:
      reconstruction_loss: 1.0
      singleton_obj_rank_loss: 1.0
      max_singleton_attr_rank_loss: 10.0  # varies in Group D
      modular_subspace_loss: 0.0
      attr_orthogonality_loss: 0.0
      attr_polarization_loss: 0.0  # varies in Group E
      concept_similarity_loss: 0.0
      union_consistency_loss: 1.0
      intersection_consistency_loss: 1.0
      attr_sink_loss: 10.0
      galois_attr_loss: 10.0
      galois_obj_loss: 1.0
      global_galois_loss: 0.0
      basis_orthogonality_loss: 20.0
      loss_obj_card_prop: 1.0
      loss_attr_card_inv_prop: 1.0
      loss_attr_obj_inv_prop: 1.0
      repulsion_loss_obj: 1.0
      repulsion_loss_attr: 1.0
      residual_orthogonality_loss: 0.0  # varies
      rank_conservation_loss: 0.0  # varies
```

## Key Metrics to Track

| Metric | What it tells us | Target direction |
|--------|-----------------|-----------------|
| attr rank card_1 | Singleton anchoring | Near 2.0 |
| attr rank card_8 | Bottom differentiation | Near 0.0 |
| attr rank range | Full utilization | 0.0-2.0 |
| obj rank range | Obj structure quality | 1.0-4.0 |
| residual_orthogonality_loss | New loss convergence | Lower = better |
| rank_conservation_loss | Rank decomposition quality | Lower = better |
| galois_attr_loss | Galois compliance | Lower = better |
| basis_orthogonality_loss | Inter-concept orthogonality | Lower = better |
| reconstruction_loss | Visual grounding | Lower = better |
| Obj heatmap cross-color | Concept discrimination | Lower = better |
| Attr heatmap shape entries | Shape discrimination | Differentiated values |

## Expected Outcomes

1. **Groups A/B**: We expect residual_orthogonality_loss=5 to be the sweet spot. Too low (1) may have little effect; too high (10) may compete with reconstruction. Rank conservation should help but be less impactful than orthogonality.

2. **Group C**: Combined losses should be strictly better than either alone, with the orthogonality component contributing more.

3. **Group D (highest priority)**: exp_63 (binary_intersection + 1x singleton + new losses) is the most important experiment. If the new losses fix the attr flattening problem while binary_intersection maintains obj rank spread, this would be the first experiment to achieve both simultaneously.

4. **Group E**: Polarization with binary_intersection is high risk/high reward. It could produce the best heatmaps ever, or it could conflict with annealed Gumbel.

## Quick Reference Table

| Exp | Group | res_ortho | rank_cons | singleton | polar | Key question |
|-----|-------|-----------|-----------|-----------|-------|-------------|
| 55 | Control | 0 | 0 | 10 | 0 | Reproducibility baseline |
| 56 | A | **1** | 0 | 10 | 0 | Mild residual orthogonality |
| 57 | A | **5** | 0 | 10 | 0 | Moderate residual orthogonality |
| 58 | A | **10** | 0 | 10 | 0 | Strong residual orthogonality |
| 59 | B | 0 | **1** | 10 | 0 | Mild rank conservation |
| 60 | B | 0 | **5** | 10 | 0 | Moderate rank conservation |
| 61 | C | **5** | **1** | 10 | 0 | Combined moderate |
| 62 | C | **10** | **5** | 10 | 0 | Combined strong |
| 63 | D | **5** | **1** | **1** | 0 | **PRIORITY**: binary_inter + 1x singleton + new losses |
| 64 | D | 0 | 0 | **1** | 0 | Control: binary_inter + 1x singleton |
| 65 | D | **5** | **1** | **5** | 0 | Intermediate: 5x singleton + new losses |
| 66 | E | 0 | 0 | 10 | **1** | Polarization + binary_intersection |
| 67 | E | **5** | **1** | 10 | **1** | Polarization + new losses |

All other loss weights identical to exp_51. All experiments: binary_intersection=true, annealed Gumbel (1->0.1), linear encoder, ambient_dim=4, 100 epochs.

## Total: 13 experiments (scl_exp_55 through scl_exp_67)
