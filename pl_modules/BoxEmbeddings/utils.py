from matplotlib import pyplot as plt
import numpy as np


def tensor_to_img(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5
    return np.clip(img, 0, 1)


def create_reconstruction_visualization(viz_datapoint, grid_size, image_size, current_epoch):
    grid_h, grid_w = grid_size
    num_patches = grid_h * grid_h

    images = viz_datapoint["images"]
    reconstructed_images = viz_datapoint["reconstructed_images"]

    original_full_image = images[-1]
    reconstructed_full_image = reconstructed_images[-1]
    original_patches = images[:-1]
    reconstructed_patches = reconstructed_images[:-1]

    total_rows = 1 + 2 * grid_h
    fig, axes = plt.subplots(total_rows, grid_w, figsize=(grid_w * 3, total_rows * 3.5))

    if total_rows == 1 and grid_w == 1:
        axes = np.array([[axes]])
    elif total_rows == 1:
        axes = axes.reshape(1, -1)
    elif grid_w == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f"Epoch {current_epoch + 1} Reconstructions", fontsize=16)

    # Row 0: Full Images
    axes[0, 0].imshow(tensor_to_img(original_full_image))
    axes[0, 0].set_title("Original Full Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tensor_to_img(reconstructed_full_image))
    axes[0, 1].set_title("Recon. Full Image")
    axes[0, 1].axis("off")
    for j in range(2, grid_w):
        axes[0, j].axis("off")  # Hide unused axes

    # Rows 1 to grid_h: Original Patches
    for i in range(grid_h):
        for j in range(grid_w):
            patch_idx = i * grid_w + j
            ax = axes[1 + i, j]
            ax.imshow(tensor_to_img(original_patches[patch_idx]))
            ax.axis("off")
            if i == 0 and j == 0:
                ax.set_title("Original Patches (VAE Inputs)")

    # Rows grid_h+1 to 2*grid_h: Reconstructed Patches
    for i in range(grid_h):
        for j in range(grid_w):
            patch_idx = i * grid_w + j
            ax = axes[1 + grid_h + i, j]
            ax.imshow(tensor_to_img(reconstructed_patches[patch_idx]))
            ax.axis("off")
            if i == 0 and j == 0:
                ax.set_title("Reconstructed Patches")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig