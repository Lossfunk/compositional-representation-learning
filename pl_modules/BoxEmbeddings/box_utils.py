import torch
from torch import nn
import torch.nn.functional as F


EPS = 1e-7


class BoxEmbeddingDistribution:
    def __init__(self, mu_min: torch.Tensor, mu_max: torch.Tensor, beta_min: torch.Tensor, beta_max: torch.Tensor):
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sample(self, epsilon=EPS):
        # Sample uniform noise (U ~ Uniform(0, 1))
        u_min = torch.rand_like(self.mu_min)
        u_max = torch.rand_like(self.mu_max)

        # Min Point Sampling (Sampled from a max-gumbel distribution)
        # Formula: mu - beta * log(-log(U))
        sample_min = self.mu_min - self.beta_min * torch.log(-torch.log(u_min + epsilon) + epsilon)

        # Max Point Sampling (Sampled from a min-gumbel distribution)
        # Formula: mu + beta * log(-log(U))
        sample_max = self.mu_max + self.beta_max * torch.log(-torch.log(u_max + epsilon) + epsilon)

        return sample_min, sample_max

def bessel_volume(box_dists: BoxEmbeddingDistribution, volume_temp: float = 1.0, log_scale: bool = True) -> torch.Tensor:
    """
    Computes the Bessel approximate volume of box distributions.
    
    Formula:
        Side_i = T * softplus( ( (mu_max_i - mu_min_i) - gamma * (beta_min_i + beta_max_i) ) / T )
        Volume = Prod(Side_i)
    
    Args:
        box_dists: The BoxEmbeddingDistribution object. Internal parameter shape: (..., D)
        volume_temp: Temperature (T) for the softplus smoothing. Lower = sharper volume.
        log_scale: If True, returns log(Volume). Recommended for loss computation.
    
    Returns:
        Tensor of shape (...,) containing the volume (or log-volume) of each box.
    """
    EULER_GAMMA = 0.57721566490153286060
    
    # 1. Calculate the expected extent of the box (Mean Max - Mean Min)
    # Shape: (..., D)
    raw_edge_len = box_dists.mu_max - box_dists.mu_min
    
    # 2. Calculate the uncertainty adjustment (2 * gamma * beta)
    # Since we have separate beta_min and beta_max, we sum them.
    # Expected Gumbel-Max mean = mu + beta*gamma
    # Expected Gumbel-Min mean = mu - beta*gamma
    # Difference = (mu_max - beta_max*gamma) - (mu_min + beta_min*gamma)
    #            = raw_edge_len - gamma * (beta_min + beta_max)
    beta_adjustment = EULER_GAMMA * (box_dists.beta_min + box_dists.beta_max)
    
    # 3. Apply Softplus with Temperature
    # T * softplus( (edge - correction) / T )
    adjusted_edge = raw_edge_len - beta_adjustment
    soft_edge_len = volume_temp * F.softplus(adjusted_edge / volume_temp)
    
    # 4. Compute Volume (Product over the last dimension D)
    if log_scale:
        # Log-Volume = Sum( log(soft_edge_len) )
        # We add a tiny epsilon to soft_edge_len inside log just in case, though softplus is positive.
        log_edges = torch.log(soft_edge_len + 1e-10)
        volume = torch.sum(log_edges, dim=-1)
    else:
        # Standard Volume = Prod(soft_edge_len)
        volume = torch.prod(soft_edge_len, dim=-1)
        
    return volume


def soft_box_weighted_intersection(box_dists: BoxEmbeddingDistribution, box_weights: torch.Tensor, eps = 1e-10):
    # box_dists will contain Box Distributions with internal shapes of (B, N, D)
    # box_weights will be a tensor of shape (B, M, N)

    box_weights_expanded = box_weights.unsqueeze(-1)

    # Expand the internal parameters to (B, 1, N, D)
    mu_min_expanded = box_dists.mu_min.unsqueeze(1)
    mu_max_expanded = box_dists.mu_max.unsqueeze(1)
    beta_min_expanded = box_dists.beta_min.unsqueeze(1)
    beta_max_expanded = box_dists.beta_max.unsqueeze(1)

    weights_sum = torch.sum(box_weights_expanded, dim=2, keepdim=True) + eps

    # Computing the beta values of the intersected box distribution. Shape: (B, M, 1, D)
    intersection_box_beta_min = torch.sum(box_weights_expanded * beta_min_expanded, dim=2, keepdim=True) / weights_sum
    intersection_box_beta_max = torch.sum(box_weights_expanded * beta_max_expanded, dim=2, keepdim=True) / weights_sum

    # Computing the mu_min values of the intersected box distribution. Shape (B, M, 1, D)
    arg_min = torch.log(box_weights_expanded + eps) + (mu_min_expanded / intersection_box_beta_min)
    intersection_box_mu_min = intersection_box_beta_min * torch.logsumexp(arg_min, dim=2, keepdim=True)

    # Computing the mu_max values of the intersected box distribution. Shape (B, M, 1, D)
    arg_max = torch.log(box_weights_expanded + eps) + (-mu_max_expanded / intersection_box_beta_max)
    intersection_box_mu_max = -intersection_box_beta_max * torch.logsumexp(arg_max, dim=2, keepdim=True)

    # Reshape intersection box parameters to get (B, M, D)
    intersection_box_mu_min = intersection_box_mu_min.squeeze(2)
    intersection_box_mu_max = intersection_box_mu_max.squeeze(2)
    intersection_box_beta_min = intersection_box_beta_min.squeeze(2)
    intersection_box_beta_max = intersection_box_beta_max.squeeze(2)

    intersection_box_dists = BoxEmbeddingDistribution(intersection_box_mu_min, intersection_box_mu_max, intersection_box_beta_min, intersection_box_beta_max)

    return intersection_box_dists


def pairwise_gumbel_intersection(box_a: BoxEmbeddingDistribution, box_b: BoxEmbeddingDistribution) -> BoxEmbeddingDistribution:
    """
    Computes the intersection of every box in A with every box in B.
    
    Args:
        box_a: Shape (Batch, Dim)
        box_b: Shape (NumClusters, Dim)
        
    Returns:
        intersection_box: Shape (Batch, NumClusters, Dim)
    """
    # 1. Broadcast Shapes
    # Box A: (Batch, 1, Dim)
    mu_min_a = box_a.mu_min.unsqueeze(1)
    mu_max_a = box_a.mu_max.unsqueeze(1)
    beta_min_a = box_a.beta_min.unsqueeze(1)
    beta_max_a = box_a.beta_max.unsqueeze(1)

    # Box B: (1, NumClusters, Dim)
    mu_min_b = box_b.mu_min.unsqueeze(0)
    mu_max_b = box_b.mu_max.unsqueeze(0)
    beta_min_b = box_b.beta_min.unsqueeze(0)
    beta_max_b = box_b.beta_max.unsqueeze(0)

    # 2. Compute Intersection Parameters
    # Beta: We approximate the intersection beta as the mean of the two betas
    # (Geometric interpretation: intersection fuzziness is average of inputs)
    new_beta_min = (beta_min_a + beta_min_b) / 2.0
    new_beta_max = (beta_max_a + beta_max_b) / 2.0

    # Mu Min: Smooth Maximum (LogSumExp)
    # Formula: beta * log( exp(mu_a/beta) + exp(mu_b/beta) )
    # Note: We use the `new_beta` for the scaling factor
    arg_min_a = mu_min_a / new_beta_min
    arg_min_b = mu_min_b / new_beta_min
    new_mu_min = new_beta_min * torch.logaddexp(arg_min_a, arg_min_b)

    # Mu Max: Smooth Minimum (-LogSumExp(-x))
    arg_max_a = -mu_max_a / new_beta_max
    arg_max_b = -mu_max_b / new_beta_max
    new_mu_max = -new_beta_max * torch.logaddexp(arg_max_a, arg_max_b)

    return BoxEmbeddingDistribution(new_mu_min, new_mu_max, new_beta_min, new_beta_max)


def gumbel_sigmoid(logits, temperature=1.0, hard=False, eps=1e-10):
    """
    Samples from a Gumbel-Sigmoid distribution.
    Args:
        logits: Unnormalized log-probabilities.
        temperature: Controls "smoothness". lower = more binary.
        hard: If True, returns strict 0s and 1s but keeps gradients (Straight-Through Estimator).
    """
    # Sample Gumbel noise
    uniforms = torch.rand_like(logits)
    gumbels = -torch.log(-torch.log(uniforms + eps) + eps)
    
    # Gumbel-Sigmoid sampling
    # Logic: sigmoid((logits + gumbel_noise) / temp)
    y_soft = torch.sigmoid((logits + gumbels) / temperature)
    
    if hard:
        # Straight-Through Estimator:
        # Forward pass is binary (0 or 1). Backward pass uses y_soft gradients.
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
        return ret
    else:
        return y_soft


class HierarchicalBoxEmbeddingsPrior(nn.Module):
    def __init__(self, boxes_per_level: list[int], embed_dim: int, beta_scale, init_config):
        super().__init__()
        self.boxes_per_level = boxes_per_level
        self.embed_dim = embed_dim
        self.beta_scale = beta_scale
        self.init_config = init_config
        self.num_levels = len(boxes_per_level)

        num_root_boxes = boxes_per_level[0]

        # self.root_mu_min = nn.Parameter(torch.rand(1, num_root_boxes, embed_dim) * 0.1)
        # self.root_mu_max = nn.Parameter(torch.rand(1, num_root_boxes, embed_dim) * 0.1 + 0.9)

        centers = torch.rand(1, num_root_boxes, embed_dim)
        min_width = 0.3
        width_variance = 0.3
        box_width = torch.rand(1, num_root_boxes, embed_dim) * width_variance + min_width
        self.root_mu_min = nn.Parameter(centers - box_width / 2)
        self.root_mu_max = nn.Parameter(centers + box_width / 2)

        # self.root_beta_min = nn.Parameter(torch.rand(1, num_root_boxes, embed_dim) * 0.02 + 0.005)
        # self.root_beta_max = nn.Parameter(torch.rand(1, num_root_boxes, embed_dim) * 0.02 + 0.005)
        self.root_beta_min = nn.Parameter(torch.ones(1, num_root_boxes, embed_dim) * -3.0)
        self.root_beta_max = nn.Parameter(torch.ones(1, num_root_boxes, embed_dim) * -3.0)

        self.adjacency_logits = nn.ParameterList()
        for level_idx in range(self.num_levels - 1):
            num_parents = self.boxes_per_level[level_idx]
            num_children = self.boxes_per_level[level_idx + 1]

            if init_config.get("sparse_init", False):

                logit_init_value = init_config.get("logit_value", 1.0)
                logits = torch.ones(num_children, num_parents) * -logit_init_value
                num_init_connections = init_config.get("init_config", num_parents // num_children)

                for child_idx in range(num_children):
                    parent_indices = torch.randperm(num_parents)[:num_init_connections]
                    logits[child_idx, parent_indices] = logit_init_value
            
            else:
                logits = torch.randn(num_children, num_parents) * 2.0 - 3.0
            
            self.adjacency_logits.append(nn.Parameter(logits))

    def get_root_boxes(self):
        
        beta_min = torch.sigmoid(self.root_beta_min) * self.beta_scale + 1e-6
        beta_max = torch.sigmoid(self.root_beta_max) * self.beta_scale + 1e-6
        # beta_min = F.softplus(self.root_beta_min) + 1e-6
        # beta_max = F.softplus(self.root_beta_max) + 1e-6

        root_box_dists = BoxEmbeddingDistribution(
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
        for level_idx in range(self.num_levels - 1):
            level_adjacency_logits = self.adjacency_logits[level_idx].unsqueeze(0)
            level_adjacency_weights = gumbel_sigmoid(level_adjacency_logits, temperature=0.5, hard=True)
            level_box_dists = soft_box_weighted_intersection(curr_box_dists, level_adjacency_weights)
            
            all_level_box_dists.append(level_box_dists)
            curr_box_dists = level_box_dists

        return all_level_box_dists