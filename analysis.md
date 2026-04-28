# Experiment Analysis Process

## Preparation Phase

### Downloading data from WandB

Use the generic download tool at `analysis/tools/download.py`. This replaces the per-comparison bespoke download scripts.

```bash
# Download specific experiments (auto-discovers latest run by name prefix)
python analysis/tools/download.py scl_exp_125 scl_exp_126

# Download a range of experiments
python analysis/tools/download.py --range scl_exp 125 130

# Download to a custom output directory (e.g., for a comparison analysis)
python analysis/tools/download.py scl_exp_125 scl_exp_126 -o analysis/comparisons/my_study

# Download only scalars (skip image artifacts)
python analysis/tools/download.py scl_exp_125 --no-images

# Force re-download (overwrite existing data)
python analysis/tools/download.py scl_exp_125 --force

# List available runs matching a pattern (useful to find run names)
python analysis/tools/download.py --list "scl_exp_12*"

# Download by exact WandB run name (when auto-discovery isn't enough)
python analysis/tools/download.py --run-name "scl_exp_125___2026-04-13__10-48-27"
```

**How it works:**
- We have an exp_id system where the name of each config file is the experiment name (e.g., `scl_exp_125`). A datetime string is appended to create the WandB run name (e.g., `scl_exp_125___2026-04-13__10-48-27`). The tool automatically finds the latest run matching a given experiment name.
- All numeric scalars are auto-discovered and downloaded (not just a hardcoded list).
- The `--range` flag works with any prefix: `scl_exp`, `mem_scl_exp`, `obj_exp`, etc.

**Output structure** (per experiment):
```
analysis/<experiment_name>/
    run_meta.json            # Run metadata (name, id, state, config)
    scalar_summary.json      # Summary stats (final, min, max, mean, std) per metric
    manifest.json            # Inventory of downloaded artifacts
    data/
        reconstruction_loss.json       # {steps: [...], values: [...]}
        rank_attr_cardinality_1.json
        ...
    images/
        reconstruction_visualization_step100.png
        lattice_inclusion_heatmap_step100.png
        ...
```

### Checkpoints

- Checkpoints are stored here: /mnt/home/ubuntu/workspace/experiment_root_dir
    - The directory structure is: `experiment_root_dir/{exp_type}/{exp_id}/{exp_id_datetime_str}/checkpoints/{checkpoint_name}`
    - Each checkpoint name has the epoch number at the end, e.g. `model-epoch=100.ckpt` means it is the checkpoint for epoch 100.

### Generating plots

Use the generic plot tool at `analysis/tools/plot.py`. Works with the standardized data format produced by the download tool.

```bash
# Quick experiment assessment — ranks + all key losses on a single figure
python analysis/tools/plot.py dashboard -e scl_exp_126 --smooth 0.93

# All active losses in a grid (one subplot per loss)
python analysis/tools/plot.py loss-grid -e scl_exp_126 --smooth 0.9

# Overlay a metric across experiments
python analysis/tools/plot.py overlay -e scl_exp_125 scl_exp_126 -m reconstruction_loss --smooth 0.9

# Compare rank evolution across experiments
python analysis/tools/plot.py rank-overlay -e scl_exp_125 scl_exp_126 -t attr --cards 1 2 4 8 --smooth 0.93

# Rank curves (attr + obj) for a single experiment
python analysis/tools/plot.py rank -e scl_exp_126 --smooth 0.93

# Multiple metrics on one plot
python analysis/tools/plot.py multi -e scl_exp_126 -m galois_attr_loss intersection_consistency_loss --smooth 0.9

# Bar chart of final values across an experiment range
python analysis/tools/plot.py trend --range scl_exp 90 99 -m reconstruction_loss

# See what metrics are available for an experiment
python analysis/tools/plot.py list-metrics -e scl_exp_98
```

**Subcommands:**
| Command | Purpose |
|---------|---------|
| `dashboard` | Combined figure: rank curves + key losses + summary table. Best for quick assessment. |
| `loss-grid` | Grid of all active losses for an experiment. |
| `overlay` | Single metric compared across experiments. |
| `rank-overlay` | Rank metric compared across experiments for specific cardinalities. |
| `rank` | Rank curves (attr/obj) for one experiment. |
| `multi` | Multiple metrics on one plot. |
| `trend` | Bar chart of a summary stat across an experiment range. |
| `list-metrics` | List all available downloaded metrics for an experiment. |

**Common flags:**
- `-e` / `--experiments`: Experiment names (e.g., `scl_exp_125 scl_exp_126`)
- `-m` / `--metric(s)`: Metric name(s) as they appear in the data files (e.g., `reconstruction_loss`)
- `--smooth`: EMA smoothing alpha (0 = none, 0.93 = heavy). Raw trace shown faintly underneath.
- `-o` / `--output`: Custom output path. Default: `analysis/<exp>/plots/` or `analysis/plots/`.
- `--base-dir`: Override the base directory for data (default: `analysis/`). Useful when data was downloaded to a comparison directory with `-o`.

**Typical workflow:**
1. Download: `python analysis/tools/download.py scl_exp_125 scl_exp_126`
2. Quick look: `python analysis/tools/plot.py dashboard -e scl_exp_126 --smooth 0.93`
3. Compare: `python analysis/tools/plot.py rank-overlay -e scl_exp_125 scl_exp_126 -t attr --cards 1 2 4 8 --smooth 0.93`
4. Deep dive: `python analysis/tools/plot.py loss-grid -e scl_exp_126 --smooth 0.9`

## Analysis Phase
- We first need to understand the objective *required conditions* that we are aiming for in this entire experiment direction overall.
    - For SubspaceConceptLattice, you can find these here: /mnt/home/ubuntu/workspace/code/compositional-representation-learning/experiments/SubspaceConceptLattice_details.md
    - For MemorySubspaceLattice, you can find these here: /mnt/home/ubuntu/workspace/code/compositional-representation-learning/experiments/MemorySubspaceLattice_details.md
- When performing analysis, you can do any of the following processes:
    - From the downloaded data from WandB, you can look at any analysis image. There are two types:
        - A plot. This can be a plot generated with any set of logged values, you may use a single value or multiple values within a single run or multiple runs. Making the right comparisons is important
        - A logged image. We log various image artifacts during training which we are downloading, looking through these, sometimes over the entire training run, is important to understand what the model is doing.
        - Ensure you are resizing these images down to a reasonable size to avoid excessive token usage.
    - You can create ad-hoc scripts which load the checkpoints and inspects the model's internal states, or model's output with some pre-determined data to verify certain hypotheses you may have or create novel observations.
    - Any other custom script you may think of to perform analysis.
- Start from a base set of observations and create meaningful comparisons and hypotheses to derive deep causal explanations for the observed behavior. There will be a recurring cycle of:
    1. Making observations
    2. Forming hypotheses
    3. Performing analyses (above) to test hypotheses
    4. Repeat
- The goal is to gain a deep understanding of the experiments. The main thing is understanding why the experiments were not able to achieve the required conditions. The deeper we understand this, the more likely we are to be able to design a new experiment which is able to achieve the required conditions.
- Maintain a set of working notes to keep track of your observations, hypotheses, and analyses. This will help you to stay organized and to avoid repeating work, and also ensure you don't miss or skip any important details.
- A helpful process is to look at comparisons where we are only changing one thing, and see how that affects the model's behavior. For example, if we are comparing two experiments, we should look at the difference in their configurations and see how that difference might explain the difference in their behavior. Sometimes, the difference between two runs may be a couple or few variables, which prevents clean causal inference, but see if you can still derive meaningful insights by making reasonable assumptions.

## Reporting Phase
- Once you have a deep understanding of the experiments, you should write a report summarizing your findings. This report should include:
    - A brief overview of the experiments
    - A summary of the required conditions
    - A summary of the analysis performed
    - A summary of the findings
    - A summary of the conclusions
    - A summary of the next steps
- It should have depth and detail with respect to the technical aspects of the experiments. An expert will be reading it, so it doesn't have to be overly simplified.