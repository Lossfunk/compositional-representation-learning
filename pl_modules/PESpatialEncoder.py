import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from models import ImageEncoder, PositionalEncoder, PermutationInvariantEncoder

class PESpatialEncoder(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.position_composition_method = config['position_composition_method']
        self.project_positional_embeddings = config['project_positional_embeddings']

        self.image_encoder = ImageEncoder(embed_dim = self.embed_dim)
        self.positional_encoder = PositionalEncoder(embed_dim = self.embed_dim)
        self.permutation_invariant_encoder = PermutationInvariantEncoder(input_dim = self.embed_dim, hidden_dim = self.hidden_dim)

    def forward(self, x):
        images = x['images'] # (batch_size, 3, H, W)
        object_masks = x['object_masks'] # (batch_size, H, W)
        batch_size = images.shape[0]

        feature_maps = self.image_encoder(images) # (batch_size, embed_dim, H, W)
        scale_factor_x = feature_maps.shape[3] / images.shape[3]
        scale_factor_y = feature_maps.shape[2] / images.shape[2]

        computed_object_representations = []
        ground_truth_object_representations = []

        for datapoint_idx in range(batch_size):
            object_mask_img = object_masks[datapoint_idx].unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            object_mask_featmap = F.interpolate(object_mask_img, scale_factor = (scale_factor_y, scale_factor_x), mode = 'nearest') # (1, 1, H, W)
            
            object_coordinates_featmap = (object_mask_featmap >= 0.5).nonzero(as_tuple=False) # (N, 2)
            object_coordinates_y_min = object_coordinates_featmap[:, 0].min()
            object_coordinates_x_min = object_coordinates_featmap[:, 1].min()
            object_coordinates_top_left = torch.stack([object_coordinates_y_min, object_coordinates_x_min], dim=1) # (1, 2)
            object_coordinates_relative = object_coordinates_featmap - object_coordinates_top_left # (N, 2)

            object_positional_embeddings = self.positional_encoder(object_coordinates_relative.unsqueeze(0).float()) # (1, N, embed_dim)
            if self.project_positional_embeddings:
                object_positional_embeddings = self.project_to_negative_hypersphere(object_positional_embeddings.squeeze(0)).unsqueeze(0) # (1, N, embed_dim)

            object_features = feature_maps[datapoint_idx][:, object_coordinates_featmap[:, 0], object_coordinates_featmap[:, 1]].permute(1, 0) # (N, embed_dim)
            object_features = self.project_to_negative_hypersphere(object_features) # (N, embed_dim)

            if self.position_composition_method == "add":
                object_lifted_features = object_features.unsqueeze(0) + object_positional_embeddings # (1, N, embed_dim)
            elif self.position_composition_method == "max":
                object_lifted_features = torch.maximum(object_features.unsqueeze(0), object_positional_embeddings) # (1, N, embed_dim)
            else:
                raise ValueError(f"Invalid position composition method: {self.position_composition_method}")

            computed_object_representation = self.permutation_invariant_encoder(object_lifted_features).squeeze(0) # (embed_dim,)
            ground_truth_object_representation  = object_lifted_features.max(dim=1).values.squeeze(0) # (embed_dim,)

            computed_object_representations.append(computed_object_representation)
            ground_truth_object_representations.append(ground_truth_object_representation)

        computed_object_representations = torch.stack(computed_object_representations) # (batch_size, embed_dim)
        ground_truth_object_representations = torch.stack(ground_truth_object_representations) # (batch_size, embed_dim)

        return computed_object_representations, ground_truth_object_representations

    def training_step(self, batch, batch_idx):
        computed_object_representations, ground_truth_object_representations = self(batch)

        loss = F.mse_loss(computed_object_representations, ground_truth_object_representations)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)

        return loss

    def project_to_negative_hypersphere(self, embeddings, eps = 1e-8):

        negative_embeddings = -torch.abs(embeddings) - eps # (batch_size, embed_dim)
        projected_embeddings = F.normalize(negative_embeddings, p=2, dim=1) # (batch_size, embed_dim)

        return projected_embeddings
