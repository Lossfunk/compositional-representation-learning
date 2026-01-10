import os
import argparse
import yaml
import re
import glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.utils import make_grid
from torch.nn import functional as F
from tqdm import tqdm

# Import your modules
from pl_modules.HierarchicalBoxVAE import HierarchicalBoxVAE, BoxDistribution

# --- Configuration ---
ARTIFACTS_CONFIG = {
    "prior_reconstructions": True,  # Changed order for preference
    "prior_embeddings": True,
    "sample_reconstructions": True,
    "sample_embeddings": True,
}

EPOCH_INTERVAL = 20

# --- Helper Functions ---

def get_sorted_checkpoints(checkpoint_dir):
    """Finds and sorts checkpoints by epoch number."""
    ckpts = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    def extract_epoch(ckpt_path):
        match = re.search(r"epoch=(\d+)", ckpt_path)
        return int(match.group(1)) if match else -1

    ckpts = sorted(ckpts, key=extract_epoch)
    filtered_ckpts = [c for c in ckpts if extract_epoch(c) % EPOCH_INTERVAL == 0 or extract_epoch(c) == 0]
    
    if ckpts and ckpts[-1] not in filtered_ckpts:
        filtered_ckpts.append(ckpts[-1])
        
    return filtered_ckpts, extract_epoch

def generate_test_samples(image_size=(64, 64)):
    """Creates Red Circle and Red Square."""
    samples = []
    masks = []
    names = []

    # 1. Red Circle
    img_circle = np.zeros([image_size[0], image_size[1], 3], dtype=np.uint8)
    mask_circle = np.zeros(image_size, dtype=np.uint8)
    center = (image_size[0]//2, image_size[1]//2)
    cv2.circle(img_circle, center, 15, (255, 0, 0), -1) 
    cv2.circle(mask_circle, center, 15, 255, -1)
    samples.append(img_circle)
    masks.append(mask_circle)
    names.append("Red Circle (R=15)")

    # 2. Red Square
    img_square = np.zeros([image_size[0], image_size[1], 3], dtype=np.uint8)
    mask_square = np.zeros(image_size, dtype=np.uint8)
    top_left = (center[0] - 15, center[1] - 15)
    bottom_right = (center[0] + 15, center[1] + 15)
    cv2.rectangle(img_square, top_left, bottom_right, (255, 0, 0), -1)
    cv2.rectangle(mask_square, top_left, bottom_right, 255, -1)
    samples.append(img_square)
    masks.append(mask_square)
    names.append("Red Square (S=30)")

    images_tensor = torch.stack([torch.from_numpy(s).permute(2, 0, 1).float() / 255.0 for s in samples])
    masks_tensor = torch.stack([torch.from_numpy(m).float() / 255.0 for m in masks])
    
    return {"images": images_tensor, "object_masks": masks_tensor, "names": names}

def to_cpu_dist(box_dist):
    """Helper to move a BoxDistribution to CPU to save GPU memory during collection."""
    return BoxDistribution(
        box_dist.mu_min.detach().cpu(),
        box_dist.mu_max.detach().cpu(),
        box_dist.beta_min.detach().cpu(),
        box_dist.beta_max.detach().cpu()
    )

def plot_box_embeddings(box_dists_list, labels, colors, title, epoch, embed_dim):
    """Plots box embeddings intervals."""
    fig, axes = plt.subplots(embed_dim, 1, figsize=(12, 2 * embed_dim), sharex=True, facecolor='#222222')
    fig.suptitle(f"{title} - Epoch {epoch}", fontsize=16, color='white')
    
    if embed_dim == 1: axes = [axes]

    for dim in range(embed_dim):
        ax = axes[dim]
        ax.set_facecolor('#222222')
        ax.set_ylabel(f"Dim {dim}", color='white', rotation=0, labelpad=20, va='center')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

        for dist_idx, dist in enumerate(box_dists_list):
            mu_min = dist.mu_min.numpy() # already on cpu
            mu_max = dist.mu_max.numpy()
            num_boxes = mu_min.shape[0]
            y_positions = np.linspace(0, 1, num_boxes)
            
            if isinstance(colors[dist_idx], str) and colors[dist_idx] == "grey":
                 c_vals = ['grey'] * num_boxes
            else:
                cmap = plt.get_cmap(colors[dist_idx])
                c_vals = cmap(np.linspace(0, 1, num_boxes))

            for i in range(num_boxes):
                mn = mu_min[i, dim]
                mx = mu_max[i, dim]
                y = y_positions[i] + (dist_idx * 1.5) 
                ax.plot([mn, mx], [y, y], color=c_vals[i], linewidth=2, alpha=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def plot_reconstruction(original, recon, patches, recon_patches, title):
    """Replicates visualization."""
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=16)

    # Convert to numpy and clip (Expects CPU tensors)
    def to_im(t): return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)

    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(to_im(original))
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(to_im(recon))
    ax2.set_title("Recon Image")
    ax2.axis("off")

    grid_in = make_grid(patches, nrow=int(np.sqrt(patches.shape[0])), padding=2)
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(to_im(grid_in))
    ax3.set_title("Input Patches")
    ax3.axis("off")

    grid_recon = make_grid(recon_patches, nrow=int(np.sqrt(recon_patches.shape[0])), padding=2)
    ax4 = plt.subplot(1, 4, 4)
    ax4.imshow(to_im(grid_recon))
    ax4.set_title("Recon Patches")
    ax4.axis("off")

    return fig

def plot_prior_grid(images, title):
    # Expects CPU tensor
    grid = make_grid(images, nrow=8, padding=2)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(grid.permute(1, 2, 0).numpy(), 0, 1))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    return fig

def add_section_header(pdf, title):
    fig = plt.figure(figsize=(11, 8.5))
    fig.clf()
    plt.text(0.5, 0.5, title, transform=fig.transFigure, ha="center", va="center", fontsize=30, weight='bold')
    pdf.savefig(fig)
    plt.close()

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument("--config_path", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--output_path", type=str, default="experiment_report.pdf", help="Output PDF path")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Data (Test Samples)
    test_data = generate_test_samples(config['model']['config']['input_resolution'])
    test_images = test_data["images"]
    test_masks = test_data["object_masks"]
    sample_names = test_data["names"]

    # 3. Get Checkpoints
    ckpts, epoch_extractor = get_sorted_checkpoints(args.checkpoint_dir)
    print(f"Found {len(ckpts)} checkpoints matching interval {EPOCH_INTERVAL}.")

    # --- Phase 1: Data Collection ---
    # We will store data in memory (CPU) to avoid reloading the model later.
    history = [] 

    print("Starting Data Collection Phase...")
    for ckpt_path in tqdm(ckpts, desc="Running Inference"):
        epoch = epoch_extractor(ckpt_path)
        
        # Load Model
        model = HierarchicalBoxVAE.load_from_checkpoint(ckpt_path, config=config)
        # model.eval()
        # model.freeze()
        
        entry = {'epoch': epoch, 'embed_dim': model.embed_dim, 'grid_size': model.grid_size}

        # 1. Collect Priors
        if ARTIFACTS_CONFIG["prior_embeddings"] or ARTIFACTS_CONFIG["prior_reconstructions"]:
            with torch.no_grad():
                priors = model.prior()
                # Move distributions to CPU immediately
                entry['prior_l0'] = to_cpu_dist(priors[0])
                entry['prior_l1'] = to_cpu_dist(priors[1])
                
                # If we need recons, generate them now to save moving model later, but store as CPU tensor
                if ARTIFACTS_CONFIG["prior_reconstructions"]:
                    z_l0 = torch.cat([priors[0].mu_min.squeeze(0), priors[0].mu_max.squeeze(0)], dim=-1)
                    z_l1 = torch.cat([priors[1].mu_min.squeeze(0), priors[1].mu_max.squeeze(0)], dim=-1)
                    entry['prior_recon_l0'] = model.vae.decode(z_l0).cpu()
                    entry['prior_recon_l1'] = model.vae.decode(z_l1).cpu()

        # 2. Collect Samples
        if ARTIFACTS_CONFIG["sample_embeddings"] or ARTIFACTS_CONFIG["sample_reconstructions"]:
            batch = {"images": test_images.to(model.device), "object_masks": test_masks.to(model.device)}
            with torch.no_grad():
                out = model(batch)
            
            # Store necessary outputs on CPU
            entry['sample_out'] = {
                'patch_box_dists': to_cpu_dist(out["patch_box_dists"]),
                'full_image_box_dists': to_cpu_dist(out["full_image_box_dists"]),
                'images': out["images"].cpu(),
                'full_image_reconstructions': out["full_image_reconstructions"].cpu(),
                'patches': out["patches"].cpu(),
                'patch_reconstructions': out["patch_reconstructions"].cpu()
            }

        history.append(entry)
        
        # Cleanup GPU
        del model
        torch.cuda.empty_cache()

    # --- Phase 2: PDF Generation ---
    print("Starting PDF Generation Phase...")
    
    with PdfPages(args.output_path) as pdf:
        
        # Title Page
        first_page = plt.figure(figsize=(11, 8.5))
        first_page.clf()
        plt.text(0.5, 0.5, f"Experiment Report\n{config['model']['type']}\n\nCheckpoints: {len(ckpts)}", 
                 transform=first_page.transFigure, ha="center", fontsize=24)
        pdf.savefig(first_page)
        plt.close()

        # Group 1: Prior Reconstructions
        if ARTIFACTS_CONFIG["prior_reconstructions"]:
            add_section_header(pdf, "Prior Reconstructions")
            for entry in tqdm(history, desc="Plotting Prior Recons"):
                fig_l0 = plot_prior_grid(entry['prior_recon_l0'], f"Prior Recon L0 (Patches) - Epoch {entry['epoch']}")
                pdf.savefig(fig_l0)
                plt.close(fig_l0)

                fig_l1 = plot_prior_grid(entry['prior_recon_l1'], f"Prior Recon L1 (Full Images) - Epoch {entry['epoch']}")
                pdf.savefig(fig_l1)
                plt.close(fig_l1)

        # Group 2: Prior Embeddings
        if ARTIFACTS_CONFIG["prior_embeddings"]:
            add_section_header(pdf, "Prior Embeddings")
            for entry in tqdm(history, desc="Plotting Prior Embeddings"):
                # Squeeze dimensions to match original logic
                l0_prior = entry['prior_l0']
                l0_prior = BoxDistribution(l0_prior.mu_min.squeeze(0), l0_prior.mu_max.squeeze(0),
                                         l0_prior.beta_min.squeeze(0), l0_prior.beta_max.squeeze(0))
                
                l1_prior = entry['prior_l1']
                l1_prior = BoxDistribution(l1_prior.mu_min.squeeze(0), l1_prior.mu_max.squeeze(0),
                                         l1_prior.beta_min.squeeze(0), l1_prior.beta_max.squeeze(0))

                fig_prior_emb = plot_box_embeddings(
                    [l0_prior, l1_prior], 
                    ["Patches (L0)", "Images (L1)"], 
                    ["viridis", "grey"], 
                    "Prior Embeddings", 
                    entry['epoch'], 
                    entry['embed_dim']
                )
                pdf.savefig(fig_prior_emb)
                plt.close(fig_prior_emb)

        # Group 3: Sample Reconstructions
        if ARTIFACTS_CONFIG["sample_reconstructions"]:
            add_section_header(pdf, "Test Sample Reconstructions")
            for entry in tqdm(history, desc="Plotting Sample Recons"):
                out = entry['sample_out']
                n_patches = entry['grid_size'][0] * entry['grid_size'][1]
                
                for i, name in enumerate(sample_names):
                    p_start = i * n_patches
                    p_end = (i + 1) * n_patches

                    fig_samp_recon = plot_reconstruction(
                        out["images"][i], 
                        out["full_image_reconstructions"][i], 
                        out["patches"][p_start:p_end], 
                        out["patch_reconstructions"][p_start:p_end],
                        f"Recon: {name} - Epoch {entry['epoch']}"
                    )
                    pdf.savefig(fig_samp_recon)
                    plt.close(fig_samp_recon)

        # Group 4: Sample Embeddings
        if ARTIFACTS_CONFIG["sample_embeddings"]:
            add_section_header(pdf, "Test Sample Embeddings")
            for entry in tqdm(history, desc="Plotting Sample Embeddings"):
                out = entry['sample_out']
                n_patches = entry['grid_size'][0] * entry['grid_size'][1]
                
                for i, name in enumerate(sample_names):
                    p_start = i * n_patches
                    p_end = (i + 1) * n_patches
                    
                    sample_patch_dists = BoxDistribution(
                        out["patch_box_dists"].mu_min[p_start:p_end],
                        out["patch_box_dists"].mu_max[p_start:p_end],
                        out["patch_box_dists"].beta_min[p_start:p_end],
                        out["patch_box_dists"].beta_max[p_start:p_end]
                    )
                    
                    sample_img_dist = BoxDistribution(
                        out["full_image_box_dists"].mu_min[i:i+1],
                        out["full_image_box_dists"].mu_max[i:i+1],
                        out["full_image_box_dists"].beta_min[i:i+1],
                        out["full_image_box_dists"].beta_max[i:i+1]
                    )

                    fig_samp_emb = plot_box_embeddings(
                        [sample_patch_dists, sample_img_dist],
                        ["Sample Patches", "Sample Image"],
                        ["inferno", "grey"], 
                        f"Embedding: {name}",
                        entry['epoch'], 
                        entry['embed_dim']
                    )
                    pdf.savefig(fig_samp_emb)
                    plt.close(fig_samp_emb)

    print(f"Report generated successfully at: {args.output_path}")

if __name__ == "__main__":
    main()