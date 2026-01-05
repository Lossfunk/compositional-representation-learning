import torch
from torch import nn
from torch.nn import functional as F

from .box_utils import BoxDistribution, soft_box_weighted_intersection, gumbel_sigmoid

class HierarchicalBoxPrior(nn.Module):
    def __init__(self, prior_config: dict, embed_dim: int, beta_scale=0.4, beta_pre_init=-3.0, beta_activation="sigmoid"):
        super().__init__()
        self.prior_config = prior_config
        self.init_config = prior_config["init_config"]
        self.boxes_per_level = prior_config["boxes_per_level"]
        self.embed_dim = embed_dim
        self.beta_scale = beta_scale
        self.beta_pre_init = beta_pre_init
        self.beta_activation = beta_activation
        self.num_levels = len(self.boxes_per_level)

        num_root_boxes = self.boxes_per_level[0]

        centers = torch.rand(1, num_root_boxes, embed_dim) * self.init_config["center_variance"] # Range: [0, center_variance]
        box_width = torch.rand(1, num_root_boxes, embed_dim) *  self.init_config["width_variance"] +  self.init_config["min_width"] # Range: [min_width, min_width + width_variance]
        # Initialize mu_min and mu_max
        self.root_mu_min = nn.Parameter(centers - box_width / 2)
        self.root_mu_max = nn.Parameter(centers + box_width / 2)

        self.root_beta_min = nn.Parameter(torch.ones(1, num_root_boxes, embed_dim) * self.beta_pre_init)
        self.root_beta_max = nn.Parameter(torch.ones(1, num_root_boxes, embed_dim) * self.beta_pre_init)

        self.adjacency_logits = nn.ParameterList()
        for level_idx in range(self.num_levels - 1):
            num_parents = self.boxes_per_level[level_idx]
            num_children = self.boxes_per_level[level_idx + 1]
            logit_init_value = self.init_config.get("logit_value", 1.0)
            
            if self.init_config.get("sparse_init", False):
                logits = torch.ones(num_children, num_parents) * -logit_init_value
                num_init_connections = self.init_config.get("num_init_connections", num_parents // num_children)

                for child_idx in range(num_children):
                    parent_indices = torch.randperm(num_parents)[:num_init_connections]
                    logits[child_idx, parent_indices] = logit_init_value
            
            else:
                logits = torch.randn(num_children, num_parents) * 2.0*logit_init_value - logit_init_value
            
            self.adjacency_logits.append(nn.Parameter(logits))

    def get_root_boxes(self):

        if self.beta_activation == "sigmoid":
            beta_min = torch.sigmoid(self.root_beta_min) * self.beta_scale + 1e-6
            beta_max = torch.sigmoid(self.root_beta_max) * self.beta_scale + 1e-6
        elif self.beta_activation == "softplus":
            beta_min = F.softplus(self.root_beta_min) + 1e-6
            beta_max = F.softplus(self.root_beta_max) + 1e-6

        root_box_dists = BoxDistribution(
            mu_min = self.root_mu_min,
            mu_max = self.root_mu_max,
            beta_min = beta_min,
            beta_max = beta_max
        ) # Internal parameters shape: (1, num_root_boxes, embed_dim)

        return root_box_dists

    def forward(self):
        root_box_dists = self.get_root_boxes()
        all_level_box_dists = [root_box_dists]
        curr_box_dists = root_box_dists

        gumbel_temp = self.prior_config.get("gumbel_temp", 0.5)
        gumbel_hard = self.prior_config.get("gumbel_hard", False)

        for level_idx in range(self.num_levels - 1):
            level_adjacency_logits = self.adjacency_logits[level_idx].unsqueeze(0)
            level_adjacency_weights = gumbel_sigmoid(level_adjacency_logits, temperature=gumbel_temp, hard=gumbel_hard)
            level_box_dists = soft_box_weighted_intersection(curr_box_dists, level_adjacency_weights)
            
            all_level_box_dists.append(level_box_dists)
            curr_box_dists = level_box_dists

        return all_level_box_dists