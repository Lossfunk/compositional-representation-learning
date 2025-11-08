import torch
import lightning as L

from models import ImageEncoder, PositionalEncoder, PermutationInvariantEncoder


class HyperNetworkSpatialEncoder(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]

        self.image_encoder = ImageEncoder(embed_dim=self.embed_dim)
        self.positional_encoder = PositionalEncoder(embed_dim=self.embed_dim)
        self.permutation_invariant_encoder = PermutationInvariantEncoder(
            input_dim=self.embed_dim, hidden_dim=self.hidden_dim
        )

    def forward(self, x):
        images = x["images"]  # (batch_size, 3, H, W)
        object_masks = x["object_masks"]  # (batch_size, H, W)
        batch_size = images.shape[0]

        feature_maps = self.image_encoder(images)  # (batch_size, embed_dim, H, W)
        scale_factor_x = feature_maps.shape[3] / images.shape[3]
        scale_factor_y = feature_maps.shape[2] / images.shape[2]

        spatial_operator_embeddings = []
        for datapoint_idx in range(batch_size):
            object_mask = object_masks[datapoint_idx]  # (H, W)
            feature_map = feature_maps[datapoint_idx]  # (embed_dim, H, W)

            object_coordinates_img = (object_mask >= 0.5).nonzero(as_tuple=False)  # (N, 2)

            object_coordinates_feat_y = (
                (object_coordinates_img[:, 0].float() * scale_factor_y).floor().long()
            )  # (N,)
            object_coordinates_feat_x = (
                (object_coordinates_img[:, 1].float() * scale_factor_x).floor().long()
            )  # (N,)

            object_coordinates_feat = torch.stack(
                [object_coordinates_feat_y, object_coordinates_feat_x], dim=1
            )  # (N, 2)
            object_coordinates_feat = torch.unique(object_coordinates_feat, dim=0)  # (M, 2)

            object_feat_y_min = object_coordinates_feat[:, 0].min()
            object_feat_x_min = object_coordinates_feat[:, 1].min()
            object_feat_top_left = torch.stack(
                [object_feat_y_min, object_feat_x_min], dim=1
            )  # (1, 2)

            object_coordinates_feat_relative = (
                object_coordinates_feat - object_feat_top_left
            )  # (M, 2)
            object_positional_embeddings = self.positional_encoder(
                object_coordinates_feat_relative.unsqueeze(0).float()
            )  # (1, M, embed_dim)

            spatial_operator_embedding = self.permutation_invariant_encoder(
                object_positional_embeddings
            ).squeeze(
                0
            )  # (embed_dim,)
            spatial_operator_embeddings.append(spatial_operator_embedding)

            object_features = feature_map[
                :, object_coordinates_feat[:, 0], object_coordinates_feat[:, 1]
            ].permute(
                1, 0
            )  # (M, embed_dim)
            object_features = (
                object_features.unsqueeze(0) + object_positional_embeddings
            )  # (1, M, embed_dim)
            object_features = object_features.squeeze(0)  # (M, embed_dim)

        spatial_operator_embeddings = torch.stack(spatial_operator_embeddings, dim=0)

        pass
