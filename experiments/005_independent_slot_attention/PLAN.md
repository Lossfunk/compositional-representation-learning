# Experiment Set 005: Independent Slot Attention & Higher-Cardinality Losses

## Motivation

Experiment set 004 (mem_scl_exp_10-20) established that:

1. **Gumbel-Softmax (exp_13, tau=1.0) achieves the best singleton attr lattice** — genuine 2-group shape-based differentiation with clean 0.95/0.0 heatmap structure. This is our new baseline.

2. **The 2-group ceiling** is the primary bottleneck: the encoder can learn one visual axis (color OR shape) but not both simultaneously because both attr slots extract features from the same shared MHA K/V projection.

3. **Soft utilization fails**: satisfies its loss degenerately without breaking attr collapse. Hard utilization is non-differentiable.

4. **Higher-cardinality representations** need attention: singleton differentiation is improving, but combination concepts (pairs, triples) lack explicit rank and structure pressure.

This experiment set introduces:

### New Architecture: Per-Slot Independent Attention
Each attr slot gets its own multi-layer transformer block (cross-attention to image patches) with independent K/V/Q projections. This lets slot 0 learn color features and slot 1 learn shape features independently.

- `per_slot_attention: true` — independent MHA per slot
- `slot_attention_depth: N` — number of cross-attention layers per slot (1 or 2)

### New Losses

1. **Rank conservation**: For each pair of singleton attr projectors, enforces `rank(intersection) + rank(residual) = rank(original)`. Adapted from SCL experiments. Ensures subspace arithmetic is well-behaved.

2. **Cosine repulsion**: Alternative to inclusion-based repulsion. Operates on basis vectors X directly (cosine similarity of flattened basis) rather than on projectors. May provide sharper gradient signal for separating proposals.

3. **Anti-collapse utilization**: Penalizes candidates with zero usage across ALL cardinalities (not just singletons). Operates on full candidate set including zero-vector options.

4. **Rank-conditioned utilization**: Within each candidate rank level, pushes for uniform usage. E.g., among concepts that should have rank 1, ensures both rank-1 candidates are used equally.

## Base Config

All experiments start from **exp_13** (best from 004):
- `resolution_method: gumbel_softmax`, `resolution_tau: 1.0`
- `utilization_mode: hard`, all structural weights = 1.0
- `embed_dim: 128`, `ambient_dim: 8`, `n_attr: 2`, `n_obj: 4`
- `fillers_per_slot: [2, 2]`, `lr: 0.0001`, `max_epochs: 100`

## Experiment Design

### Phase 1: Per-Slot Attention Ablation (exp_21-24)

| Exp | per_slot_attention | slot_attention_depth | Other Changes | Key Question |
|-----|-------------------|---------------------|---------------|-------------|
| **21** | true | 2 | none | Does independent 2-layer slot attention break the 2-group ceiling? |
| **22** | true | 1 | none | Depth 1 vs 2: is multi-layer necessary or is K/V independence sufficient? |
| **23** | true | 2 | cosine_repulsion=1, inclusion_repulsion=0 | Does cosine repulsion work better than inclusion repulsion with per-slot attention? |
| **24** | true | 2 | cosine_repulsion=1, inclusion_repulsion=1 | Both repulsion types combined — does it help? |

### Phase 2: Higher-Cardinality Losses (exp_25-28)

All use per_slot_attention=true, slot_attention_depth=2 (from Phase 1).

| Exp | rank_conservation | anti_collapse_util | rank_cond_util | Key Question |
|-----|------------------|-------------------|---------------|-------------|
| **25** | 1.0 | 0 | 0 | Does rank conservation alone improve higher-cardinality structure? |
| **26** | 0 | 1.0 (attr+obj) | 0 | Does anti-collapse utilization push non-singleton diversity? |
| **27** | 0 | 0 | 1.0 (attr+obj) | Does rank-conditioned utilization improve structure? |
| **28** | 1.0 | 1.0 (attr+obj) | 1.0 (attr+obj) | Kitchen sink: all new losses together |

### Phase 3: Combined Best (exp_29-30)

| Exp | Description |
|-----|-------------|
| **29** | Best repulsion from Phase 1 + best cardinality losses from Phase 2 |
| **30** | exp_29 + loss_targets=proposal for new losses (structural losses on proposals instead of resolved) |

Note: exp_29 and exp_30 configs will be created after Phase 1 and 2 results are analyzed.

## Key Metrics

### Singleton differentiation (primary, from 004)
- Unique attr candidates used (target: 4/4)
- Attr inclusion heatmap quality (target: 4-way differentiation)
- Proposal cosine similarity across images (target: break +/-1.0 degeneracy)

### Higher-cardinality (new focus)
- Attr rank spread: card_1 - card_8 (target: ~1.905 - 0 = 1.905)
- Attr rank monotonicity: card_1 > card_2 > card_4 > card_8
- Intersection consistency loss (target: lower than exp_13's)
- Rank conservation loss value

### Overall lattice quality
- Galois attr + obj (target: both near 0)
- Reconstruction loss (target: comparable to exp_13)
- Inverse proportionality (target: near 0)

## Hypotheses

1. Per-slot attention (exp_21/22) will break the 2-group ceiling and achieve 3-4 unique attr candidates, because independent K/V projections allow each slot to specialize on different visual features.
2. Depth 2 (exp_21) will outperform depth 1 (exp_22), as iterative refinement allows more complex feature extraction.
3. Cosine repulsion (exp_23) will provide stronger separation than inclusion repulsion at the proposal level, since it directly operates on basis vectors rather than through the projector abstraction.
4. Rank conservation (exp_25) will improve attr rank ordering across cardinalities without hurting singleton structure.
5. The kitchen sink (exp_28) will achieve the best overall lattice structure, as the new losses target complementary aspects of the representation.

## Code Changes (already implemented)

1. **`ConceptEncoder.py`**: `per_slot_attention` config flag, `slot_attention_depth` config, multi-layer cross-attention per slot with LayerNorm + residual connections.
2. **`VQSubspaceConceptLattice.py`**:
   - `_rank_conservation_loss`: projector-based intersection + residual rank conservation
   - `_compute_cosine_repulsion`: cosine similarity on flattened basis vectors
   - `_utilization_loss_anti_collapse`: penalizes zero-usage candidates across all cardinalities
   - `_utilization_loss_rank_conditioned`: within-rank-level uniform usage
   - All new losses are config-gated (default weight 0) and backward compatible

## Run Command

```bash
./run_experiments.sh --type mem_scl --range 21 28 --max-parallel 4
```
