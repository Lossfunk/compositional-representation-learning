Analyze a MemorySubspaceConceptLattice experiment. The user will provide an experiment name (e.g., "mem_scl_exp_0") or a full run path.

## Steps

### 1. Locate the experiment
- Look in `/mnt/home/ubuntu/workspace/experiment_root_dir/MemorySubspaceConceptLattice/{exp_name}/`
- Use the most recent run (sorted by timestamp in directory name) unless the user specifies otherwise
- Also check the wandb directory at `./wandb/` for the matching run

### 2. Read scalar summary
- Check for `wandb-summary.json` in the wandb run's `files/` directory
- Extract key metrics at final epoch:
  - **Loss values**: total_loss, reconstruction_loss, commitment_loss, galois_attr/obj, intersection_consistency, union_consistency, attr_sink, inv_prop, memory_slot_ortho, memory_obj_rank, memory_obj_ortho, repulsion_attr/obj, utilization
  - **Rank metrics**: rank/attr_cardinality_1 through rank/attr_cardinality_B, same for obj
  - **Training info**: epoch count, tau value
- Flag any loss that dominates total_loss (>50% of total)

### 3. Inspect visualizations
- Look for images in the wandb run's `files/media/images/` directory
- **IMPORTANT**: Downsize all images before reading to save context tokens:
  ```python
  from PIL import Image
  img = Image.open(path)
  w, h = img.size
  scale = min(600/w, 600/h, 1.0)
  img.resize((int(w*scale), int(h*scale)), Image.LANCZOS).save('/tmp/..._small.png')
  ```
- Read the downsized versions of:
  - `lattice_inclusion_heatmap_*` (latest epoch) — shows obj and attr inclusion structure
  - `memory_entries_*` (latest epoch) — shows memory codebook values
  - `reconstruction_visualization_*` (latest epoch) — shows reconstruction quality
- Key things to look for in heatmaps:
  - **Obj inclusion**: Do singletons (C5-C8) have high inclusion in their correct higher concepts (C1-C4, C0)? Do color/shape groupings emerge?
  - **Attr inclusion**: Are values differentiated across concepts or uniform (degenerate)?
  - **Memory entries title**: Shows assignment patterns (which singleton maps to which memory candidate)

### 4. Load checkpoint and inspect internals (if needed)
- Load the last checkpoint from `{run_dir}/checkpoints/`
- Check critical internal quantities:
  - `concept_encoder.obj_memory` — per-entry row norms (should be ~1.0 from QR init)
  - `concept_encoder.attr_memory[s]` — per-filler norms
  - **Proposal output scale**: Run a forward pass on a batch and check `raw_X_attr` and `raw_X_obj` norms. Scale mismatch between proposals and memory causes commitment loss explosion.
- Use this pattern:
  ```python
  import torch, yaml, sys
  sys.path.insert(0, '.')
  from pl_modules.SubspaceConceptLattice.VQSubspaceConceptLattice import MemorySubspaceConceptLattice
  from datasets import get_dataset

  with open(config_path) as f: config = yaml.safe_load(f)
  model = MemorySubspaceConceptLattice(config)
  ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
  model.load_state_dict(ckpt['state_dict'])
  model.eval()
  ```

### 5. Diagnosis checklist
For each issue, check:
- **Commitment loss explosion**: Are proposal norms >> memory norms? (attr proposals should match attr_memory scale ~0.1-1.0; obj proposals should match obj_memory scale ~1.0)
- **Memory collapse**: Do all singletons pick the same memory candidate? (check assignment indices in memory_entries title)
- **Attr degenerate**: Are all attr inclusion values uniform (e.g., all 0.68)? This means attr subspaces aren't differentiating.
- **Obj structure**: Does the obj heatmap show correct block structure? (singletons high in their concepts, low elsewhere)
- **Reconstruction quality**: Are reconstructions recognizable? Color and shape correct?
- **Rank ordering**: Do attr ranks decrease and obj ranks increase with cardinality?

### 6. Report format
Summarize findings as:
```
## Experiment: {name}
### Config: {key config diffs from baseline}
### Key Metrics (epoch {N}):
- total_loss: X (dominated by: Y)
- reconstruction: X
- commitment: X
- [other notable losses]
- attr ranks: card_1=X, card_B=X (spread: X)
- obj ranks: card_1=X, card_B=X (spread: X)

### Diagnosis:
[Main finding and root cause]

### Visualizations:
[Brief description of heatmap/recon quality]

### Recommendations:
[Proposed fixes]
```

## Environment
- Python: `/home/ubuntu/miniconda3/envs/vh-crl/bin/python`
- Working dir: `/mnt/home/ubuntu/workspace/code/compositional-representation-learning`
- Experiments: `/mnt/home/ubuntu/workspace/experiment_root_dir/MemorySubspaceConceptLattice/`
