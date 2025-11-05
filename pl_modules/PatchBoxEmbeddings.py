import math
import itertools
import random

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from torchvision import transforms
from vector_quantize_pytorch import VectorQuantize

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, stride = 1, padding = "same")
        self.conv_2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = "same")

        self.maxpool_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_3 = nn.Conv2d(in_channels = 8, out_channels = 12, kernel_size = 3, stride = 1, padding = "same")
        self.conv_4 = nn.Conv2d(in_channels = 12, out_channels = 12, kernel_size = 3, stride = 1, padding = "same")

        self.maxpool_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_5 = nn.Conv2d(in_channels = 12, out_channels = 20, kernel_size = 3, stride = 1, padding = "same")
        self.conv_6 = nn.Conv2d(in_channels = 20, out_channels = 20, kernel_size = 3, stride = 1, padding = "same")

        self.maxpool_3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_7 = nn.Conv2d(in_channels = 20, out_channels = 32, kernel_size = 3, stride = 1, padding = "same")
        self.conv_8 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = "same")

        self.maxpool_4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_9 = nn.Conv2d(in_channels = 32, out_channels = 48, kernel_size = 3, stride = 1, padding = "same")
        self.conv_10 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 1, stride = 1, padding = "same")

        self.maxpool_5 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_11 = nn.Conv2d(in_channels = 48, out_channels = 64, kernel_size = 3, stride = 1, padding = "same")
        self.conv_12 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = "same")

        self.maxpool_6 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_13 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1, padding = "same")
        self.conv_14 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 1, stride = 1, padding = "same")

        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(96 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

        self.layers = nn.Sequential(
            self.conv_1,
            nn.ReLU(),
            self.conv_2,
            nn.ReLU(),
            self.maxpool_1,
            nn.ReLU(),
            self.conv_3,
            nn.ReLU(),
            self.conv_4,
            nn.ReLU(),
            self.maxpool_2,
            nn.ReLU(),
            self.conv_5,
            nn.ReLU(),
            self.conv_6,
            nn.ReLU(),
            self.maxpool_3,
            nn.ReLU(),
            self.conv_7,
            nn.ReLU(),
            self.conv_8,
            nn.ReLU(),
            self.maxpool_4,
            nn.ReLU(),
            self.conv_9,
            nn.ReLU(),
            self.conv_10,
            nn.ReLU(),
            self.maxpool_5,
            nn.ReLU(),
            self.conv_11,
            nn.ReLU(),
            self.conv_12,
            nn.ReLU(),
            self.maxpool_6,
            nn.ReLU(),
            self.conv_13,
            nn.ReLU(),
            self.conv_14,
            self.flatten,
            self.mlp
        )

    def forward(self, x):

        x = self.layers(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class BindingFunction(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class PatchBoxEmbeddings(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config['model']['config']['embed_dim']
        self.box_embed_dim = self.embed_dim * 2
        self.hidden_dim = config['model']['config']['hidden_dim']
        self.grid_size = config['model']['config']['grid_size']
        self.permutation_samples = 5

        self.image_encoder = ImageEncoder(embed_dim = self.box_embed_dim)
        self.positional_encoder = PositionalEncoder(input_dim = 2, output_dim = self.box_embed_dim, hidden_dim = self.hidden_dim)
        self.binding_function = BindingFunction(input_dim = self.box_embed_dim * 2, output_dim = self.box_embed_dim, hidden_dim = self.hidden_dim)

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.config['data']['train']['config']['image_size'][:2])
        ])

        self.vq = VectorQuantize(
            dim = self.box_embed_dim,
            codebook_size = 512,
            decay = 0.8,
            commitment_weight = 1.
        )

    def forward(self, x):
        images = x['images'] # (batch_size, 3, H, W)
        object_masks = x['object_masks'] # (batch_size, H, W)
        batch_size = images.shape[0]

        batch_image_patches = self.divide_image_into_patches(images) # (batch_size, grid_h, grid_w, C, patch_h, patch_w)
        batch_mask_patches = self.divide_image_into_patches(object_masks.unsqueeze(1)) # (batch_size, grid_h, grid_w, 1, patch_h, patch_w)
        batch_bounding_boxes = self.compute_bounding_boxes(batch_mask_patches) # (batch_size, 4)

        batch_loss_tensors = {
            "codebook_positive_volume_loss": torch.tensor(0.0, device = images.device),
            "pos_commitment_loss": torch.tensor(0.0, device = images.device),
            "patch_commitment_loss": torch.tensor(0.0, device = images.device),
            "object_region_commitment_loss": torch.tensor(0.0, device = images.device),
            "parent_commitment_loss": torch.tensor(0.0, device = images.device),
            "foundational_embeddings_separation_loss": torch.tensor(0.0, device = images.device),
            "parent_inclusion_loss": torch.tensor(0.0, device = images.device),
            "parent_permutation_inclusion_loss": torch.tensor(0.0, device = images.device),
            "parent_permutation_separation_loss": torch.tensor(0.0, device = images.device),
            "object_region_alignment_loss": torch.tensor(0.0, device = images.device),
        }

        codebook_embeddings = self.vq.codebook # (codebook_size, box_embed_dim)
        codebook_positive_volume_loss = self.compute_positive_volume_loss(codebook_embeddings) # scalar
        batch_loss_tensors["codebook_positive_volume_loss"] = codebook_positive_volume_loss

        for datapoint_idx in range(batch_size):
            image_patches = batch_image_patches[datapoint_idx] # (grid_h, grid_w, C, patch_h, patch_w)
            mask_patches = batch_mask_patches[datapoint_idx] # (grid_h, grid_w, 1, patch_h, patch_w)
            bounding_box = batch_bounding_boxes[datapoint_idx] # (4,)

            # Get the patches that contain the object
            object_patches = image_patches[bounding_box[0]:bounding_box[2] + 1, bounding_box[1]:bounding_box[3] + 1] # (bbox_grid_h, bbox_grid_w, C, patch_h, patch_w)
            object_region = self.stitch_object_patches(object_patches) # (C, stitched_h, stitched_w)
            bbox_grid_h, bbox_grid_w, C, patch_h, patch_w = object_patches.shape
            num_patches = bbox_grid_h * bbox_grid_w
            object_patches_flattened = object_patches.reshape(num_patches, C, patch_h, patch_w) # (num_patches, C, patch_h, patch_w)

            if num_patches <= 2:
                continue

            object_patches_resized = F.interpolate(object_patches_flattened, size = (224, 224), mode = "bilinear", align_corners = False) # (num_patches, C, 224, 224)
            object_region_resized = F.interpolate(object_region.unsqueeze(0), size = (224, 224), mode = "bilinear", align_corners = False) # (1, C, 224, 224)
            encoder_output = self.image_encoder(torch.cat([object_patches_resized, object_region_resized], dim = 0)) # (num_patches + 1, box_embed_dim)

            object_patch_embeddings = encoder_output[:-1] # (num_patches, box_embed_dim)
            object_region_embedding = encoder_output[-1].unsqueeze(0) # (1, box_embed_dim)

            object_grid_coords_normalized = self.compute_positional_data(object_patches) # (num_patches, 2)
            object_positional_embeddings = self.positional_encoder(object_grid_coords_normalized) # (num_patches, box_embed_dim)

            quantized_patch_embeddings, _, patch_commitment_loss = self.vq(object_patch_embeddings) # (num_patches, box_embed_dim)
            quantized_positional_embeddings, _, positional_commitment_loss = self.vq(object_positional_embeddings) # (num_patches, box_embed_dim)
            quantized_object_region_embedding, _, region_commitment_loss = self.vq(object_region_embedding) # (1, box_embed_dim)
            parent_embeddings, parent_commitment_loss = self.compute_parent_embeddings(quantized_patch_embeddings, quantized_positional_embeddings) # (num_patches, num_patches, box_embed_dim)

            #TODO: Move this loss computation below
            foundational_embeddings_separation_loss = self.compute_foundational_embeddings_loss(quantized_patch_embeddings, quantized_positional_embeddings)
            
            parent_ground_truth_boxes = self.compute_pairwise_intersection_boxes(quantized_patch_embeddings, quantized_positional_embeddings) # (num_patches, num_patches, box_embed_dim)
            parent_index_pair_inclusion_loss = self.compute_index_pair_inclusion_loss(
                parent_embeddings.reshape(num_patches * num_patches, -1),
                parent_ground_truth_boxes.reshape(num_patches * num_patches, -1)
            ) # (num_patches * num_patches,)

            #TODO: Move this loss computation below
            parent_inclusion_loss = parent_index_pair_inclusion_loss.mean()

            overall_intersection_box = self.compute_overall_intersection_box(torch.cat([quantized_positional_embeddings, quantized_patch_embeddings], dim = 0)) # (1, box_embed_dim)

            permutation_set_indices = self.compute_permutation_indices(num_patches) # (num_permutations_sets, num_patches, 2)
            num_permutation_sets = permutation_set_indices.shape[0]
            sampled_parent_permutation_sets = parent_embeddings[permutation_set_indices[..., 0], permutation_set_indices[..., 1]] # (num_permutations_sets, num_patches, box_embed_dim)

            sampled_permutation_intersection_boxes = []
            for permutation_idx in range(num_permutation_sets):
                permutation_set = sampled_parent_permutation_sets[permutation_idx] # (num_patches, box_embed_dim)
                permutation_intersection_box = self.compute_overall_intersection_box(permutation_set) # (1, box_embed_dim)
                sampled_permutation_intersection_boxes.append(permutation_intersection_box)
            
            sampled_permutation_intersection_boxes = torch.cat(sampled_permutation_intersection_boxes, dim = 0) # (num_permutation_sets, box_embed_dim)

            # TODO: Move this loss computation below
            parent_permutation_inclusion_loss = self.compute_index_pair_inclusion_loss(
                sampled_permutation_intersection_boxes,
                overall_intersection_box.expand_as(sampled_permutation_intersection_boxes)
            ) # (num_permutation_sets,)
            parent_permutation_inclusion_loss = parent_permutation_inclusion_loss.mean()

            parent_permutation_pairwise_intersection_boxes = self.compute_pairwise_intersection_boxes(sampled_permutation_intersection_boxes, sampled_permutation_intersection_boxes) # (num_permutation_sets, num_permutation_sets, box_embed_dim)
            parent_permutation_pairwise_volumes = self.compute_box_volume(parent_permutation_pairwise_intersection_boxes.reshape(-1, self.box_embed_dim)) # (num_permutation_sets * num_permutation_sets,)
            parent_permutation_pairwise_volumes = parent_permutation_pairwise_volumes.reshape(num_permutation_sets, num_permutation_sets) # (num_permutation_sets, num_permutation_sets)
            parent_permutation_pairwise_volumes_upper = parent_permutation_pairwise_volumes.triu(diagonal=1)
            parent_permutation_separation_loss = torch.sum(parent_permutation_pairwise_volumes_upper) / (num_permutation_sets * (num_permutation_sets - 1) + 1e-6) # scalar

            correct_parent_set = torch.diagonal(parent_embeddings, 0).permute(1, 0) # (num_patches, box_embed_dim)
            correct_parent_intersection_box = self.compute_overall_intersection_box(correct_parent_set) # (1, box_embed_dim)
            object_region_alignment_loss = F.mse_loss(quantized_object_region_embedding, correct_parent_intersection_box) # scalar

            batch_loss_tensors["pos_commitment_loss"] += positional_commitment_loss.mean()
            batch_loss_tensors["patch_commitment_loss"] += patch_commitment_loss.mean()
            batch_loss_tensors["object_region_commitment_loss"] += region_commitment_loss.mean()
            batch_loss_tensors["parent_commitment_loss"] += parent_commitment_loss.mean()
            batch_loss_tensors["foundational_embeddings_separation_loss"] += foundational_embeddings_separation_loss
            batch_loss_tensors["parent_inclusion_loss"] += parent_inclusion_loss
            batch_loss_tensors["parent_permutation_inclusion_loss"] += parent_permutation_inclusion_loss
            batch_loss_tensors["parent_permutation_separation_loss"] += parent_permutation_separation_loss
            batch_loss_tensors["object_region_alignment_loss"] += object_region_alignment_loss

        return batch_loss_tensors

    def training_step(self, batch, batch_idx):
        batch_loss_tensors = self.forward(batch)
        total_loss = sum(batch_loss_tensors.values())
        self.log("train_loss", total_loss, prog_bar=True, on_epoch=True)
        self.log_dict(batch_loss_tensors, prog_bar=True, on_epoch=True)
        
        return total_loss

    def configure_optimizers(self):

        if self.config['trainer']['optimizer']['type'] == 'Adam':
            return torch.optim.Adam(self.parameters(), **self.config['trainer']['optimizer']['config'])
        else:
            raise ValueError(f"Optimizer type {self.config['trainer']['optimizer']['type']} not implemented.")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            min_embed, max_embed = self.split_box_embeddings(self.vq.codebook)
            clamped_max = torch.max(max_embed, min_embed + 1e-4)
            self.vq.codebook.data[:, self.embed_dim:] = clamped_max

    def divide_image_into_patches(self, images):
        B, C, H, W = images.shape
        
        grid_h, grid_w = self.grid_size
        patch_h = H // grid_h
        patch_w = W // grid_w

        if H % grid_h != 0 or W % grid_w != 0:
            raise ValueError("Image dimensions must be divisible by grid size.")

        patches = images.view(B, C, grid_h, patch_h, grid_w, patch_w) # (B, C, grid_h, patch_h, grid_w, patch_w)
        patches = patches.permute(0, 2, 4, 1, 3, 5) # (B, grid_h, grid_w, C, patch_h, patch_w)
        patches = patches.contiguous()
        
        return patches

    def compute_bounding_boxes(self, mask_patches):
        B, grid_h, grid_w, _, patch_h, patch_w = mask_patches.shape
        mask_patches_sum = mask_patches.sum(dim = (-1, -2, -3)) # (B, grid_h, grid_w)
        non_empty_patches = mask_patches_sum > 0 # (B, grid_h, grid_w)

        y_coords = torch.arange(grid_h, device = mask_patches.device).reshape(1, grid_h, 1).expand(B, grid_h, grid_w) # (B, grid_h, grid_w)
        x_coords = torch.arange(grid_w, device = mask_patches.device).reshape(1, 1, grid_w).expand(B, grid_h, grid_w) # (B, grid_h, grid_w)

        y_coords_masked_min = torch.where(non_empty_patches, y_coords, grid_h + 1)
        x_coords_masked_min = torch.where(non_empty_patches, x_coords, grid_w + 1)
        y_coords_masked_max = torch.where(non_empty_patches, y_coords, -1)
        x_coords_masked_max = torch.where(non_empty_patches, x_coords, -1)

        y_min = y_coords_masked_min.reshape(B, -1).min(dim=1).values
        x_min = x_coords_masked_min.reshape(B, -1).min(dim=1).values
        y_max = y_coords_masked_max.reshape(B, -1).max(dim=1).values
        x_max = x_coords_masked_max.reshape(B, -1).max(dim=1).values

        y_min = torch.clamp(y_min, 0, grid_h - 1)
        x_min = torch.clamp(x_min, 0, grid_w - 1)
        y_max = torch.clamp(y_max, -1, grid_h - 1)
        x_max = torch.clamp(x_max, -1, grid_w - 1)

        return torch.stack([y_min, x_min, y_max, x_max], dim = -1) # (B, 4)

    def stitch_object_patches(self, object_patches):
        bbox_grid_h, bbox_grid_w, C, patch_h, patch_w = object_patches.shape
        stitched_h = bbox_grid_h * patch_h
        stitched_w = bbox_grid_w * patch_w
        stitched_image = object_patches.permute(2, 0, 3, 1, 4) # (C, bbox_grid_h, patch_h, bbox_grid_w, patch_w)
        stitched_image = stitched_image.reshape(C, stitched_h, stitched_w) # (C, stitched_h, stitched_w)
        
        return stitched_image

    def compute_positional_data(self, object_patches):
        bbox_grid_h, bbox_grid_w = object_patches.shape[:2]
        object_grid_coords_y = torch.arange(bbox_grid_h, device=object_patches.device, dtype=torch.float32)
        object_grid_coords_x = torch.arange(bbox_grid_w, device=object_patches.device, dtype=torch.float32)
        object_grid_coords_y, object_grid_coords_x = torch.meshgrid(object_grid_coords_y, object_grid_coords_x, indexing = "ij") # (bbox_grid_h, bbox_grid_w)
        object_grid_coords = torch.stack([object_grid_coords_y, object_grid_coords_x], dim = -1) # (bbox_grid_h, bbox_grid_w, 2)
        object_grid_coords_normalized = object_grid_coords / torch.tensor([bbox_grid_h - 1, bbox_grid_w - 1], device = object_patches.device) # (bbox_grid_h, bbox_grid_w, 2)
        object_grid_coords_normalized = object_grid_coords_normalized.reshape(bbox_grid_h * bbox_grid_w, 2) # (bbox_grid_h * bbox_grid_w, 2)

        return object_grid_coords_normalized

    def split_box_embeddings(self, embeddings):
        return embeddings[:, :self.embed_dim], embeddings[:, self.embed_dim:]

    def compute_pairwise_intersection_boxes(self, box_embeddings_1, box_embeddings_2):
        boxes_min_1, boxes_max_1 = self.split_box_embeddings(box_embeddings_1) # (N, embed_dim), (N, embed_dim)
        boxes_min_2, boxes_max_2 = self.split_box_embeddings(box_embeddings_2) # (N, embed_dim), (N, embed_dim)

        boxes_min_1_i = boxes_min_1.unsqueeze(1) # (N, 1, embed_dim)
        boxes_min_2_j = boxes_min_2.unsqueeze(0) # (1, N, embed_dim)
        boxes_max_1_i = boxes_max_1.unsqueeze(1) # (N, 1, embed_dim)
        boxes_max_2_j = boxes_max_2.unsqueeze(0) # (1, N, embed_dim)
        
        intersection_min = torch.max(boxes_min_1_i, boxes_min_2_j) # (N, N, embed_dim)
        intersection_max = torch.min(boxes_max_1_i, boxes_max_2_j) # (N, N, embed_dim)
        intersection_boxes = torch.cat([intersection_min, intersection_max], dim=-1) # (N, N, box_embed_dim)

        return intersection_boxes

    def compute_index_pair_inclusion_loss(self, box_embeddings_1, box_embeddings_2):
        boxes_min_1, boxes_max_1 = self.split_box_embeddings(box_embeddings_1) # (N, embed_dim), (N, embed_dim)
        boxes_min_2, boxes_max_2 = self.split_box_embeddings(box_embeddings_2) # (N, embed_dim), (N, embed_dim)

        min_violation = F.relu(boxes_min_1 - boxes_min_2) # (N, embed_dim)
        max_violation = F.relu(boxes_max_2 - boxes_max_1) # (N, embed_dim)
        loss = min_violation.sum(dim=-1) + max_violation.sum(dim=-1) # (N,)
        
        return loss

    def compute_overall_intersection_box(self, box_embeddings):
        boxes_min, boxes_max = self.split_box_embeddings(box_embeddings) # (N, embed_dim), (N, embed_dim)
        intersection_min = torch.max(boxes_min, dim = 0).values.unsqueeze(0) # (1, embed_dim)
        intersection_max = torch.min(boxes_max, dim = 0).values.unsqueeze(0) # (1, embed_dim)
        intersection_box = torch.cat([intersection_min, intersection_max], dim = -1) # (1, box_embed_dim)
        
        return intersection_box

    def compute_box_volume(self, box_embeddings):
        boxes_min, boxes_max = self.split_box_embeddings(box_embeddings) # (N, embed_dim), (N, embed_dim)
        box_sizes = torch.clamp(boxes_max - boxes_min, min = 0) # (N, embed_dim)
        # box_volume = torch.prod(box_sizes, dim=-1) # (N,)
        log_volume = torch.sum(torch.log(box_sizes + 1e-6), dim=-1) # (N,)
        
        return log_volume

    def compute_positive_volume_loss(self, box_embeddings, min_side = 0.1):
        num_boxes = box_embeddings.shape[0]
        boxes_min, boxes_max = self.split_box_embeddings(box_embeddings)
        box_sizes = boxes_max - boxes_min
        loss = F.relu(min_side - box_sizes)
        loss = torch.sum(loss) / num_boxes

        return torch.sum(loss)

    def compute_permutation_indices(self, num_elements):
        # permutations = list(itertools.permutations(range(num_elements)))
        # import ipdb; ipdb.set_trace()
        # num_permutations_sets = min(self.permutation_samples, len(permutations))
        # permutations_sampled = random.sample(permutations, num_permutations_sets)
        total_possible_permutations = math.factorial(num_elements)
        num_permutations_sets = min(self.permutation_samples, total_possible_permutations)
        permutation_sets = []
        permutation_sets_filled = False
        while not permutation_sets_filled:
            base_list = list(range(num_elements))
            random.shuffle(base_list)
            if base_list not in permutation_sets:
                permutation_sets.append(base_list)
            if len(permutation_sets) == num_permutations_sets:
                permutation_sets_filled = True

        cols = torch.tensor(permutation_sets, dtype=torch.long) # (num_permutations_sets, num_elements)
        rows = torch.arange(num_elements, dtype=torch.long).unsqueeze(0).expand(num_permutations_sets, -1) # (num_permutations_sets, num_elements)
        indices = torch.stack([rows, cols], dim=-1) # (num_permutations_sets, num_elements, 2)

        return indices

    # def compute_foundational_embeddings_loss(self, quantized_patch_embeddings, quantized_positional_embeddings):
    #     unique_quantized_patch_embeddings, _ = torch.unique(quantized_patch_embeddings, dim=0, return_inverse=True) # (num_unique, box_embed_dim), (N)
    #     foundational_embeddings = torch.cat([quantized_positional_embeddings, unique_quantized_patch_embeddings], dim = 0) # (N + num_unique, box_embed_dim)
    #     foundational_embeddings_min, foundational_embeddings_max = self.split_box_embeddings(foundational_embeddings) # (N + num_unique, embed_dim), (N + num_unique, embed_dim)
    #     foundational_embeddings_centers = (foundational_embeddings_min + foundational_embeddings_max) / 2.0 # (N + num_unique, embed_dim)
    #     foundational_embeddings_distance_margin = 0.1
    #     foundational_embeddings_pairwise_distances = torch.pdist(foundational_embeddings_centers, p=2) # (N + num_unique, N + num_unique)
    #     foundational_embeddings_separation_loss = F.relu(foundational_embeddings_distance_margin - foundational_embeddings_pairwise_distances) # (N + num_unique, N + num_unique)
    #     foundational_embeddings_separation_loss = torch.sum(foundational_embeddings_separation_loss) / (foundational_embeddings_centers.shape[0] * (foundational_embeddings_centers.shape[0] - 1)) # scalar

    #     return foundational_embeddings_separation_loss

    def compute_foundational_embeddings_loss(self, quantized_patch_embeddings, quantized_positional_embeddings):
        N, M = quantized_positional_embeddings.shape[0], quantized_patch_embeddings.shape[0]
        foundational_embeddings = torch.cat([quantized_positional_embeddings, quantized_patch_embeddings], dim = 0) # (N + num_unique, box_embed_dim)
        foundational_embeddings_min, foundational_embeddings_max = self.split_box_embeddings(foundational_embeddings) # (N + num_unique, embed_dim), (N + num_unique, embed_dim)
        foundational_embeddings_centers = (foundational_embeddings_min + foundational_embeddings_max) / 2.0 # (N + num_unique, embed_dim)
        
        foundational_embeddings_distance_margin = 0.1
        foundational_embeddings_pairwise_distances = torch.cdist(foundational_embeddings_centers, foundational_embeddings_centers, p=2) # (N + num_unique, N + num_unique)
        foundational_embeddings_separation_loss = F.relu(foundational_embeddings_distance_margin - foundational_embeddings_pairwise_distances) # (N + num_unique, N + num_unique)
        
        mask = torch.ones_like(foundational_embeddings_pairwise_distances, dtype=torch.bool)
        mask.fill_diagonal_(False)
        patch_eq_matrix = torch.all(
            quantized_patch_embeddings.unsqueeze(1) == quantized_patch_embeddings.unsqueeze(0),
            dim=-1
        ) 
        mask[N:, N:] = ~patch_eq_matrix
        mask = mask.float()

        foundational_embeddings_separation_loss = foundational_embeddings_separation_loss * mask
        foundational_embeddings_separation_loss = foundational_embeddings_separation_loss.sum() / (mask.sum() + 1e-6)

        return foundational_embeddings_separation_loss

    def compute_parent_embeddings(self, quantized_patch_embeddings, quantized_positional_embeddings):
        num_patches = quantized_patch_embeddings.shape[0]
        pos_expand = quantized_positional_embeddings.unsqueeze(1).expand(-1, num_patches, -1) # (N, N, box_embed_dim)
        patch_expand = quantized_patch_embeddings.unsqueeze(0).expand(num_patches, -1, -1) # (N, N, box_embed_dim)
        dense_concat = torch.cat([pos_expand, patch_expand], dim=-1) # (N, N, box_embed_dim * 2)
        parent_embeddings = self.binding_function(dense_concat.reshape(num_patches * num_patches, -1)) # (N * N, box_embed_dim)
        parent_embeddings, _, parent_commitment_loss = self.vq(parent_embeddings) # (N * N, box_embed_dim)
        parent_embeddings = parent_embeddings.reshape(num_patches, num_patches, -1) # (N, N, box_embed_dim)

        return parent_embeddings, parent_commitment_loss