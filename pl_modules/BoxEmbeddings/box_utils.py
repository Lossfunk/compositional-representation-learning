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
    
    Source: "Improving Local Identifiability in Probabilistic Box Embeddings", Dasgupta et al. 2020.
    As described in "Box Embeddings" library paper, Table 4[cite: 153].
    
    Formula:
        Side_i = T * softplus( ( (mu_max_i - mu_min_i) - gamma * (beta_min_i + beta_max_i) ) / T )
        Volume = Prod(Side_i)
    
    Args:
        box_dists: The BoxEmbeddingDistribution object.
        volume_temp: Temperature (T) for the softplus smoothing. Lower = sharper volume.
        log_scale: If True, returns log(Volume). Recommended for loss computation.
    
    Returns:
        Tensor of shape (Batch, ...) containing the volume (or log-volume) of each box.
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
    def __init__(self, boxes_per_level: list[int], embed_dim: int):
        super().__init__()
        self.boxes_per_level = boxes_per_level
        self.embed_dim = embed_dim
        self.num_levels = len(boxes_per_level)

        num_root_boxes = boxes_per_level[0]
        self.root_mu_min = nn.Parameter(torch.rand(1, num_root_boxes, embed_dim) * 0.5)
        root_mu_delta = torch.rand(1, num_root_boxes, embed_dim) * 0.5 + 0.5
        # self.root_mu_max = nn.Parameter(torch.rand(1, num_root_boxes, embed_dim) * 0.4 + 0.6)
        self.root_mu_max = nn.Parameter(self.root_mu_min + root_mu_delta)
        # self.root_beta_min = nn.Parameter(torch.rand(1, num_root_boxes, embed_dim) * 0.02 + 0.005)
        # self.root_beta_max = nn.Parameter(torch.rand(1, num_root_boxes, embed_dim) * 0.02 + 0.005)
        self.root_beta_min = nn.Parameter(torch.ones(1, num_root_boxes, embed_dim) * -3.0)
        self.root_beta_max = nn.Parameter(torch.ones(1, num_root_boxes, embed_dim) * -3.0)

        self.adjacency_logits = nn.ParameterList()
        for level_idx in range(self.num_levels - 1):
            num_parents = self.boxes_per_level[level_idx]
            num_children = self.boxes_per_level[level_idx + 1]

            # logits = torch.randn(num_children, num_parents)
            logits = torch.randn(num_children, num_parents) * 2.0 - 3.0
            self.adjacency_logits.append(nn.Parameter(logits))

    def get_root_boxes(self):
        
        beta_scale = 0.05
        beta_min = torch.sigmoid(self.root_beta_min) * beta_scale + 1e-6
        beta_max = torch.sigmoid(self.root_beta_max) * beta_scale + 1e-6

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
            level_adjacency_weights = gumbel_sigmoid(level_adjacency_logits, temperature=0.1, hard=False)
            level_box_dists = soft_box_weighted_intersection(curr_box_dists, level_adjacency_weights)
            
            all_level_box_dists.append(level_box_dists)
            curr_box_dists = level_box_dists

        return all_level_box_dists


# def soft_volume(
#     box_dist: BoxEmbeddingDistribution,
#     euler_gamma: float = 0.57721566490153286060,
#     volume_temp = 1.0,
#     log_scale: bool = True
# ):
    
#     exact_side_lengths = (box_dist.mu_max - box_dist.mu_min) - euler_gamma * (box_dist.beta_min + box_dist.beta_max)
#     soft_side_lengths = volume_temp * F.softplus(exact_side_lengths / volume_temp)

#     if log_scale:
#         log_vol = torch.sum(torch.log(soft_side_lengths + 1e-20), dim=-1)
#         return log_vol
#     else:
#         vol = torch.prod(soft_side_lengths, dim=-1)
#         return vol


# def soft_intersection(
#     box_dist: BoxEmbeddingDistribution,
#     box_weights: torch.Tensor,
# ):
#     """
#     Computes the soft intersection of a batch of box distributions.
    
#     Args:
#         box_dist: A BoxEmbeddingDistribution where tensors have shape (N, ...).
#                   The intersection is computed over the dimension 0 (size N).
#         box_weights: Tensor of shape (N, ...) representing weights for each box.
#                      Should be broadcastable to the shape of box_dist parameters 
#                      (except dim 0 which it matches).
#     """
    
#     # Extract parameters (N, ...)
#     boxes_mu_min = box_dist.mu_min
#     boxes_mu_max = box_dist.mu_max
#     boxes_beta_min = box_dist.beta_min
#     boxes_beta_max = box_dist.beta_max

#     # Ensure weights are broadcastable
#     # If weights are 1D (N,), reshape to (N, 1, 1, ...) to match rank
#     while box_weights.ndim < boxes_mu_min.ndim:
#         box_weights = box_weights.unsqueeze(-1)
        
#     log_weights = torch.log(box_weights + 1e-20)

#     # --- 1. Compute Intersection Beta (Scale) ---
#     # Formula: 1 / Sum( w_i / beta_i ) 
#     # (Weighted Harmonic Mean logic for Gumbel scale)

#     # Min Beta
#     term_min = box_weights / (boxes_beta_min + 1e-20)
#     sum_term_min = torch.sum(term_min, dim=0)
#     intersection_beta_min = 1.0 / (sum_term_min + 1e-20)

#     # Max Beta
#     term_max = box_weights / (boxes_beta_max + 1e-20)
#     sum_term_max = torch.sum(term_max, dim=0)
#     intersection_beta_max = 1.0 / (sum_term_max + 1e-20)

#     # --- 2. Compute Intersection Mu (Location) ---
#     # Formula: beta_new * LogSumExp( log(w_i) + mu_i / beta_i )
#     # (Product of Experts logic for Gumbel location)

#     # Min Mu
#     boxes_mu_min_scaled = boxes_mu_min / (boxes_beta_min + 1e-20)
#     # LogSumExp over dim 0
#     lse_min = torch.logsumexp(log_weights + boxes_mu_min_scaled, dim=0)
#     intersection_mu_min = intersection_beta_min * lse_min

#     # Max Mu
#     # Note: Max point follows Min-Gumbel, so we work with negations
#     boxes_mu_max_scaled = -boxes_mu_max / (boxes_beta_max + 1e-20)
#     lse_max = torch.logsumexp(log_weights + boxes_mu_max_scaled, dim=0)
#     intersection_mu_max = -intersection_beta_max * lse_max

#     return BoxEmbeddingDistribution(
#         mu_min=intersection_mu_min,
#         mu_max=intersection_mu_max,
#         beta_min=intersection_beta_min,
#         beta_max=intersection_beta_max
#     )


def elementwise_gumbel_intersection(
    dist_A: BoxEmbeddingDistribution, 
    dist_B: BoxEmbeddingDistribution
) -> BoxEmbeddingDistribution:
    """
    Computes intersection between box A_i and box B_i for all i.
    Shapes must be identical (or broadcastable without unsqueezing).
    """
    # Formula: 1 / (1/beta_A + 1/beta_B)
    sum_inv_beta_min = (1.0 / (dist_A.beta_min + 1e-20)) + (1.0 / (dist_B.beta_min + 1e-20))
    int_beta_min = 1.0 / (sum_inv_beta_min + 1e-20)

    sum_inv_beta_max = (1.0 / (dist_A.beta_max + 1e-20)) + (1.0 / (dist_B.beta_max + 1e-20))
    int_beta_max = 1.0 / (sum_inv_beta_max + 1e-20)

    # Mu Min
    term_min_A = dist_A.mu_min / (dist_A.beta_min + 1e-20)
    term_min_B = dist_B.mu_min / (dist_B.beta_min + 1e-20)
    int_mu_min = int_beta_min * (term_min_A + term_min_B)

    # Mu Max
    term_max_A = -dist_A.mu_max / (dist_A.beta_max + 1e-20)
    term_max_B = -dist_B.mu_max / (dist_B.beta_max + 1e-20)
    int_mu_max = -int_beta_max * (term_max_A + term_max_B)

    return BoxEmbeddingDistribution(int_mu_min, int_mu_max, int_beta_min, int_beta_max)


# class HierarchicalBoxEmbeddingsPrior(nn.Module):
#     def __init__(self, boxes_per_level: list[int], embed_dim: int):
#         super().__init__()
#         self.boxes_per_level = boxes_per_level
#         self.embed_dim = embed_dim
#         self.num_levels = len(boxes_per_level)
        
#         num_root_boxes = boxes_per_level[0]
#         # self.root_mu_min = nn.Parameter(torch.zeros(num_root_boxes, embed_dim))
#         # self.root_mu_delta = nn.Parameter(torch.ones(num_root_boxes, embed_dim) * 0.5)

#         # self.root_beta_min = nn.Parameter(torch.ones(num_root_boxes, embed_dim))
#         # self.root_beta_max = nn.Parameter(torch.ones(num_root_boxes, embed_dim))

#         # 1. Initialize mu_min in range [-2, 2]
#         # Sigmoid(-2) ≈ 0.12, Sigmoid(2) ≈ 0.88
#         # This covers most of the space without saturating gradients at extreme -5/5.
#         self.root_mu_min = nn.Parameter(torch.rand(num_root_boxes, embed_dim) * 4.0 - 2.0)

#         # 2. Initialize mu_delta in range [-2, 2]
#         # This allows for both very small boxes (delta fraction ~0.1) and large boxes (delta fraction ~0.9)
#         self.root_mu_delta = nn.Parameter(torch.rand(num_root_boxes, embed_dim) * 4.0 - 2.0)

#         # 3. Initialize Beta (Uncertainty) small
#         # We want initial priors to be somewhat "sharp" so they are distinct.
#         # Initialize raw parameters such that softplus(param) is small (e.g. 0.05).
#         # softplus(-3.0) ≈ 0.05
#         self.root_beta_min = nn.Parameter(torch.randn(num_root_boxes, embed_dim) - 3.0)
#         self.root_beta_max = nn.Parameter(torch.randn(num_root_boxes, embed_dim) - 3.0)

#         self.adjacency_weights = nn.ParameterList()

#         for level_idx in range(self.num_levels - 1):
#             num_parents = self.boxes_per_level[level_idx]
#             num_children = self.boxes_per_level[level_idx + 1]

#             weights = torch.randn(num_children, num_parents)
#             self.adjacency_weights.append(nn.Parameter(weights))

#     def get_root_boxes(self):
#         mu_min = torch.sigmoid(self.root_mu_min)
#         # mu_max = mu_min + F.softplus(self.root_mu_delta)
#         delta_fraction = torch.sigmoid(self.root_mu_delta)
#         mu_max = mu_min + (1.0 - mu_min) * delta_fraction
#         # beta_min = F.softplus(self.root_beta_min)
#         # beta_max = F.softplus(self.root_beta_max)

#         beta_scale = 0.05 
#         beta_min = torch.sigmoid(self.root_beta_min) * beta_scale + 1e-6
#         beta_max = torch.sigmoid(self.root_beta_max) * beta_scale + 1e-6
        
#         return BoxEmbeddingDistribution(mu_min, mu_max, beta_min, beta_max)

#     def batched_soft_intersection(
#         self, 
#         parent_dist: BoxEmbeddingDistribution, 
#         adj_logits: torch.Tensor
#     ) -> BoxEmbeddingDistribution:

#         weights = F.softmax(adj_logits, dim=1) # (M, N)
#         weights_expanded = weights.unsqueeze(-1) # (M, N, 1)
#         log_weights = torch.log(weights_expanded + 1e-20)
        
#         p_mu_min = parent_dist.mu_min.unsqueeze(0)       # (1, N, D)
#         p_mu_max = parent_dist.mu_max.unsqueeze(0)       # (1, N, D)
#         p_beta_min = parent_dist.beta_min.unsqueeze(0)   # (1, N, D)
#         p_beta_max = parent_dist.beta_max.unsqueeze(0)   # (1, N, D)
        
#         # Min Beta
#         term_min = weights_expanded / (p_beta_min + 1e-20) # (M, N, D)
#         sum_term_min = torch.sum(term_min, dim=1)          # (M, D) -- Sum over parents
#         c_beta_min = 1.0 / (sum_term_min + 1e-20)

#         # Max Beta
#         term_max = weights_expanded / (p_beta_max + 1e-20)
#         sum_term_max = torch.sum(term_max, dim=1)
#         c_beta_max = 1.0 / (sum_term_max + 1e-20)
        
#         # Min Mu (Max-Gumbel Logic)
#         scaled_mu_min = p_mu_min / (p_beta_min + 1e-20)
#         lse_min = torch.logsumexp(log_weights + scaled_mu_min, dim=1) # Sum over parents
#         c_mu_min = c_beta_min * lse_min

#         # Max Mu (Min-Gumbel Logic -> Negate)
#         scaled_mu_max = -p_mu_max / (p_beta_max + 1e-20)
#         lse_max = torch.logsumexp(log_weights + scaled_mu_max, dim=1) # Sum over parents
#         c_mu_max = -c_beta_max * lse_max
        
#         return BoxEmbeddingDistribution(c_mu_min, c_mu_max, c_beta_min, c_beta_max)

#     def forward(self) -> list[BoxEmbeddingDistribution]:

#         root_boxes = self.get_root_boxes()
#         all_box_levels = [root_boxes]
        
#         # Iteratively compute next levels
#         curr_box_level = root_boxes
#         for weights in self.adjacency_weights:
#             next_box_level = self.batched_soft_intersection(curr_box_level, weights)
#             all_box_levels.append(next_box_level)
#             curr_box_level = next_box_level
            
#         return all_box_levels



# class BoxTensor:
#     def __init__(self, min_embed: torch.Tensor, max_embed: torch.Tensor):
#         self.min_embed = min_embed
#         self.max_embed = max_embed

#     @property
#     def shape(self):
#         return self.min_embed.shape

#     def center(self):
#         return (self.min_embed + self.max_embed) / 2.0

#     def side_lengths(self):
#         return F.softplus(self.max_embed - self.min_embed)

#     def __repr__(self):
#         return f"BoxTensor(shape={self.min_embed.shape})"


# def gumbel_intersection(
#     box_1: BoxTensor, box_2: BoxTensor, temp: torch.Tensor or float
# ) -> BoxTensor:
#     """
#     Computes the soft intersection of two boxes.
#     Supports broadcasting (e.g. box_1 [B, 1, D] and box_2 [1, K, D] -> [B, K, D]).
#     """
#     box_1_min, box_1_max = box_1.min_embed, box_1.max_embed
#     box_2_min, box_2_max = box_2.min_embed, box_2.max_embed

#     # 1. Soft Max for Intersection Min
#     # Formula: temp * log( exp(a/temp) + exp(b/temp) )
#     # torch.logaddexp(x, y) = log(exp(x) + exp(y)) and handles broadcasting automatically
#     intersection_min_embed = temp * torch.logaddexp(
#         box_1_min / temp, 
#         box_2_min / temp
#     )

#     # 2. Soft Min for Intersection Max
#     # Formula: -temp * log( exp(-a/temp) + exp(-b/temp) )
#     intersection_max_embed = -temp * torch.logaddexp(
#         -box_1_max / temp, 
#         -box_2_max / temp
#     )

#     return BoxTensor(intersection_min_embed, intersection_max_embed)


# def soft_volume(box: BoxTensor, temp: torch.Tensor or float) -> torch.Tensor:
#     """
#     Computes the log soft volume of the box.
#     Manual implementation of softplus to support tensor temperature.
#     Formula: temp * log(1 + exp(x / temp))
#     """
#     # Calculate raw side lengths
#     diff = box.max_embed - box.min_embed
    
#     # softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
#     # Here beta = 1/temp. So: temp * log(1 + exp(diff / temp))
    
#     # We use logaddexp for numerical stability:
#     # log(1 + exp(y)) = log(exp(0) + exp(y)) = logaddexp(0, y)
    
#     # 1. Scale difference by temperature
#     scaled_diff = diff / temp
    
#     # 2. Compute soft side lengths in log space first to be safe
#     # soft_length = temp * log(1 + exp(scaled_diff))
#     soft_length = temp * torch.logaddexp(torch.tensor(0.0, device=diff.device), scaled_diff)
    
#     # 3. Compute Log Volume (Sum of log side lengths)
#     # LogVol = Sum( log(soft_length) )
#     # Note: soft_length is already positive.
#     log_vol = torch.sum(torch.log(soft_length + EPS), dim=-1)

#     return log_vol


# class HierarchicalPrior(nn.Module):
#     def __init__(self, boxes_per_level, embed_dim, temp=1.0):
#         super().__init__()

#         self.boxes_per_level = boxes_per_level
#         self.embed_dim = embed_dim
#         self.num_levels = len(boxes_per_level)

#         self.intersection_temp = nn.Parameter(torch.tensor(float(temp)))

#         num_root_boxes = boxes_per_level[0]
#         self.root_min_embed = nn.Parameter(torch.rand(num_root_boxes, embed_dim) * 0.2 + 0.3)
#         self.root_delta_embed = nn.Parameter(torch.rand(num_root_boxes, embed_dim) * 0.3 + 0.2)

#         self.adj_params = nn.ParameterList()
#         for i in range(self.num_levels - 1):
#             num_parents = boxes_per_level[i]
#             num_children = boxes_per_level[i + 1]
#             self.adj_params.append(nn.Parameter(torch.randn(num_children, num_parents)))

#     def get_root_boxes(self):
#         root_max = self.root_min_embed + F.softplus(self.root_delta_embed)
#         return BoxTensor(self.root_min_embed, root_max)

#     def compute_adj_matrix(self, level_idx, hard=False):
#         logits = self.adj_params[level_idx - 1]

#         if self.training and not hard:
#             uniform = torch.rand_like(logits)
#             gumbel_noise = -torch.log(-torch.log(uniform + EPS) + EPS)
#             adj = torch.sigmoid(logits + gumbel_noise)
#         else:
#             adj = torch.sigmoid(logits)
#             if hard:
#                 adj = (adj > 0.5).float()

#         return adj

#     def compute_level_boxes(self, parent_boxes: BoxTensor, adj_matrix: torch.Tensor):
#         parent_min = parent_boxes.min_embed.unsqueeze(0)
#         parent_max = parent_boxes.max_embed.unsqueeze(0)

#         w = adj_matrix.unsqueeze(-1)

#         large_val = 1e4

#         parent_min_masked = parent_min * w + (1 - w) * (-large_val)
#         parent_max_masked = parent_max * w + (1 - w) * (large_val)

#         temp = self.intersection_temp

#         child_min_embed = temp * torch.logsumexp(parent_min_masked / temp, dim=1)
#         child_max_embed = -temp * torch.logsumexp(-parent_max_masked / temp, dim=1)

#         return BoxTensor(child_min_embed, child_max_embed)

#     def forward(self, hard_adjacency=False):
#         all_levels_boxes = [self.get_root_boxes()]
#         all_adj_matrices = []

#         for level_idx in range(1, self.num_levels):
#             adj_matrix = self.compute_adj_matrix(level_idx, hard=hard_adjacency)
#             level_boxes = self.compute_level_boxes(all_levels_boxes[-1], adj_matrix)

#             all_adj_matrices.append(adj_matrix)
#             all_levels_boxes.append(level_boxes)

#         return all_levels_boxes, all_adj_matrices
