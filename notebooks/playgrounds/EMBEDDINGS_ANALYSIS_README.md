# Box Embeddings Analysis - Implementation Guide

## Overview

This implementation provides a complete pipeline for analyzing box embeddings from the PatchBoxEmbeddingsVAE model, including custom data generation, embedding extraction, and multiple visualization styles.

## Files Created

- **`PatchBoxEmbeddingsVAE_embeddings_analysis.ipynb`**: Main notebook with all functionality

## Key Components

### 1. Custom Data Generation

**Function**: `generate_custom_datapoints(param_dicts, config)`

Creates custom datapoints from parameter specifications:

```python
param_dicts = [
    {
        'shape': 'circle',      # 'circle' or 'square'
        'color': 'red',         # 'red' or 'blue'
        'size': 20,             # radius for circle, side_length for square
        'position': (32, 32)    # (x, y) - center for circle, top-left for square
    },
    # ... more datapoints
]

custom_datapoints = generate_custom_datapoints(param_dicts, config)
```

### 2. Embedding Extraction

**Function**: `extract_embeddings(model, datapoints)`

Extracts box embeddings (min, max coordinates) for:
- All patch embeddings: `(num_patches, 2, embed_dim)`
- Full image embedding: `(2, embed_dim)`

```python
embedding_results = extract_embeddings(model, custom_datapoints)

# Access embeddings
patch_boxes = embedding_results[0]['patch_boxes']  # All patch boxes
image_box = embedding_results[0]['image_box']      # Full image box
metadata = embedding_results[0]['metadata']        # Original parameters
```

### 3. Visualization Functions

#### a) Detailed Per-Dimension View

**Function**: `visualize_box_embeddings(embedding_results, sample_idx=0)`

- Shows each dimension in a separate subplot
- Displays intervals [min, max] for full image and all patches
- Color-coded patches with legend
- Best for detailed dimension-by-dimension analysis

#### b) Compact Stacked View

**Function**: `visualize_box_embeddings_compact(embedding_results, sample_idx=0)`

- Single chart with stacked bars
- Shows cumulative extents across dimensions
- Black background style similar to the reference image
- Orange (min extent) and teal (interval extent) colors
- Best for quick overview

#### c) Dimension-wise Interval View (Recommended)

**Function**: `visualize_dimension_intervals(embedding_results, sample_idx=0)`

- Shows all embeddings overlaid for each dimension
- Full image as thick bar, patches as thin bars with offsets
- Dark theme with color-coded embeddings
- Best for comparing patch boxes to image box across dimensions

## Usage Example

```python
# 1. Define custom datapoints
param_dicts = [
    {'shape': 'circle', 'color': 'red', 'size': 20, 'position': (32, 32)},
    {'shape': 'square', 'color': 'blue', 'size': 25, 'position': (50, 50)},
]

# 2. Generate datapoints
custom_datapoints = generate_custom_datapoints(param_dicts, config)

# 3. Extract embeddings
embedding_results = extract_embeddings(model, custom_datapoints)

# 4. Visualize
visualize_dimension_intervals(embedding_results, sample_idx=0)

# Or visualize all samples
for idx in range(len(embedding_results)):
    visualize_dimension_intervals(embedding_results, sample_idx=idx)
```

## Understanding Box Embeddings

Box embeddings represent each patch and the full image as **hyper-rectangles** in a high-dimensional space:

- **min**: Lower bounds of the box in each dimension (embed_dim,)
- **max**: Upper bounds of the box in each dimension (embed_dim,)
- **interval**: Width = max - min in each dimension

### Interpretation

1. **Full Image Box**: Represents the entire image content in embedding space
2. **Patch Boxes**: Represent individual patches of the image
3. **Containment**: Ideally, patch boxes should be "contained" within or overlap with the image box
4. **Dimension Analysis**: Different dimensions capture different aspects of the visual content

### Visualization Features

- **Interval Bars**: Show the [min, max] range for each dimension
- **Overlap**: Visual indication of how patch boxes relate to image box
- **Zero Line**: Red dashed line shows the origin (0) in each dimension
- **Grid**: Helps read exact values from the visualization

## Tips

1. **Start with dimension intervals view** for best overview of box relationships
2. **Use detailed view** when investigating specific dimensions
3. **Compare multiple samples** to understand model behavior
4. **Look for patterns** in which dimensions have large/small intervals
5. **Check containment** - are patches within the image box bounds?

## Extension Ideas

- Add metrics for box containment (IoU, Jaccard similarity)
- Visualize box volume across dimensions
- Compare embeddings for similar vs different shapes/colors
- Analyze how patch position affects embeddings
- Plot 2D projections of box embeddings (PCA, t-SNE)


