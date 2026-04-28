# Experiment Set 006: Obj Subspace Isolation

## Goal
Establish that the obj subspace pathway works correctly in isolation before tackling the full attr+obj lattice. The obj pathway is NOT used for reconstruction — all structure must come from structural + memory + commitment losses.

## Expected behavior
With B=4 images, n_obj=4:
- 4 singletons → 4 distinct rank-1 obj subspaces (each maps to a different memory entry)
- 6 pairs → rank-2 unions
- 4 triples → rank-3 unions
- 1 quad → rank-4 (full space)
- Galois: each singleton's obj ⊆ its parent concept's obj
- Union consistency: combined obj = ridge_projector(union of singleton bases)

## Losses enabled
All attr-related losses are OFF (weight=0). Only obj-relevant losses:
- `memory_obj_rank_loss` — push each memory entry to rank-1
- `memory_obj_orthogonality_loss` — memory entries mutually orthogonal
- `commitment_loss` — VQ-VAE style proposal ↔ memory alignment
- `repulsion_loss_obj` / `cosine_repulsion_loss_obj` — push singleton projectors apart
- `galois_obj_loss` — singleton ⊆ combined
- `union_consistency_loss` — combined = union of singleton bases
- `utilization_loss_obj` — soft, singleton candidates all used
- `utilization_anti_collapse_obj` — soft, all-cardinality candidates used

## Ablation dimensions
1. **Resolution temperature** (`resolution_tau`): 1.0, 0.5, 0.1
2. **Obj rank loss mode** (`obj_rank_loss_mode`): "svd" vs "trace"
3. **Repulsion mode**: inclusion only, cosine only, both
4. **Structural weights**: 1x vs 10x on galois_obj + union_consistency
5. **Key loss ablation**: remove galois, remove union consistency, remove utilization
6. **Rank-conditioned utilization**: on/off

## Experiment matrix

| Exp | Description | resolution_tau | rank_mode | repulsion | galois_obj | union_consist | util_obj | anti_collapse_obj | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **Baseline** | 1.0 | svd | inclusion=1 | 1.0 | 1.0 | 1.0 | 1.0 | Reference |
| 1 | Lower gumbel temp | 0.5 | svd | inclusion=1 | 1.0 | 1.0 | 1.0 | 1.0 | Sharper selection |
| 2 | Near-hard gumbel | 0.1 | svd | inclusion=1 | 1.0 | 1.0 | 1.0 | 1.0 | Near-deterministic |
| 3 | Trace rank loss | 1.0 | trace | inclusion=1 | 1.0 | 1.0 | 1.0 | 1.0 | Same metric as logging |
| 4 | Trace + low temp | 0.5 | trace | inclusion=1 | 1.0 | 1.0 | 1.0 | 1.0 | Best of 3+1? |
| 5 | Cosine repulsion only | 1.0 | svd | cosine=1 | 1.0 | 1.0 | 1.0 | 1.0 | Per-basis-vector |
| 6 | Both repulsions | 1.0 | svd | incl+cos=1 | 1.0 | 1.0 | 1.0 | 1.0 | Complementary? |
| 7 | High structural wts | 1.0 | svd | inclusion=1 | 10.0 | 10.0 | 1.0 | 1.0 | Stronger pressure |
| 8 | No galois_obj | 1.0 | svd | inclusion=1 | 0.0 | 1.0 | 1.0 | 1.0 | Is Galois needed? |
| 9 | No union_consistency | 1.0 | svd | inclusion=1 | 1.0 | 0.0 | 1.0 | 1.0 | Is UC needed? |
| 10 | No utilization | 1.0 | svd | inclusion=1 | 1.0 | 1.0 | 0.0 | 0.0 | Collapse to 1 entry? |
| 11 | + rank_cond_obj | 1.0 | svd | inclusion=1 | 1.0 | 1.0 | 1.0 | 1.0 | rank_cond_obj=1 |

Exp 12 (best-of combo) deferred until reviewing 0-11.

## Key metrics
- obj_rank per cardinality: expect card_k → k
- galois_obj_loss → 0
- union_consistency_loss → 0
- utilization_loss_obj → low (all entries used)
- concept_ranks visualization: clean staircase for obj
- obj heatmap: 4 distinct singleton columns, correct unions

## Config prefix
`configs/MemorySubspaceConceptLattice/mem_scl_obj_exp_{N}.yaml`
