import torch
import torch.nn.functional as F

EPS = 1e-10

class BoxDistribution:
    def __init__(self, mu_min: torch.Tensor, mu_max: torch.Tensor, beta_min: torch.Tensor, beta_max: torch.Tensor):
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sample(self, epsilon=EPS):
        # Sample uniform noise (U ~ Uniform(0, 1))
        u_min = torch.rand_like(self.mu_min)
        u_max = torch.rand_like(self.mu_max)

        # Min Point Sampling (Max-gumbel distribution) (mu - beta * log(-log(U)))
        sample_min = self.mu_min - self.beta_min * torch.log(-torch.log(u_min + epsilon) + epsilon)

        # Max Point Sampling (Min-gumbel distribution) (mu + beta * log(-log(U)))
        sample_max = self.mu_max + self.beta_max * torch.log(-torch.log(u_max + epsilon) + epsilon)

        return sample_min, sample_max


def bessel_volume(box_dists: BoxDistribution, volume_temp: float = 1.0, log_scale: bool = True, eps: float = EPS) -> torch.Tensor:
    """
    Computes the Bessel approximate volume of box distributions.
    
    Args:
        box_dists: The BoxDistribution object. Internal parameter shape: (..., D)
        volume_temp: Temperature (T) for the softplus smoothing. Lower = sharper volume.
        log_scale: If True, returns log(Volume). Recommended for loss computation.

    Formula:
        Side_i = T * softplus(((mu_max_i - mu_min_i) - gamma * (beta_min_i + beta_max_i))/T)
        Volume = Prod(Side_i)
    
    Returns:
        Tensor of shape (...,) containing the volume (or log-volume) of each box.
    """
    EULER_GAMMA = 0.57721566490153286060
    
    # Calculate the expected extent of the box (Mean Max - Mean Min)
    raw_edge_len = box_dists.mu_max - box_dists.mu_min # (..., D)
    
    # Calculate the uncertainty adjustment (2 * gamma * beta)
    beta_adjustment = EULER_GAMMA * (box_dists.beta_min + box_dists.beta_max) # (..., D)
    
    # Apply Softplus with Temperature (T * softplus((edge - correction) / T))
    adjusted_edge = raw_edge_len - beta_adjustment # (..., D)
    soft_edge_len = volume_temp * F.softplus(adjusted_edge / volume_temp) # (..., D)
    
    # Compute Volume (Product over the last dimension D)
    if log_scale:
        # Log-Volume = Sum(log(soft_edge_len))
        log_edges = torch.log(soft_edge_len + eps) # (..., D)
        volume = torch.sum(log_edges, dim=-1) # (...,)
    else:
        # Standard Volume = Prod(soft_edge_len)
        volume = torch.prod(soft_edge_len, dim=-1) # (...,)
        
    return volume


def soft_box_weighted_intersection(box_dists: BoxDistribution, box_weights: torch.Tensor, eps: float = EPS) -> BoxDistribution:
    """
    Computes weighted intersections of box distributions.

    Args:
        box_dists: The BoxDistribution object. Internal parameter shape: (B, N, D)
        box_weights: The intersection weights. Shape: (B, M, N)

    N -> Number of input boxes
    M -> Number of output boxes
    box_weights is essentially a weighted adjacency matrix between the input box dists and output intersected box dists.

    Formula:
        Let w_i be the box_weights for the i-th input box.
        Let (mu_min_i, mu_max_i, beta_min_i, beta_max_i) be the parameters of the i-th input box.
        
        1. Weighted Uncertainty (Beta Interpolation):
           beta_min_new = sum(w_i * beta_min_i) / sum(w_i)
           beta_max_new = sum(w_i * beta_max_i) / sum(w_i)

        2. Weighted Min Coordinate (Weighted SoftMax):
           mu_min_new = beta_min_new * log(sum(w_i * exp(mu_min_i / beta_min_new)))

        3. Weighted Max Coordinate (Weighted SoftMin):
           mu_max_new = -beta_max_new * log(sum(w_i * exp(-mu_max_i / beta_max_new)))

    Returns:
        intersection_box_dists: A BoxDistribution object containing intersections of the input box_dists with respect to the box_weights. Internal parameter shape: (B, M, D)
    """

    box_weights_expanded = box_weights.unsqueeze(-1) # (B, M, N, 1)

    # Expand the internal parameters to (B, 1, N, D)
    mu_min_expanded = box_dists.mu_min.unsqueeze(1)
    mu_max_expanded = box_dists.mu_max.unsqueeze(1)
    beta_min_expanded = box_dists.beta_min.unsqueeze(1)
    beta_max_expanded = box_dists.beta_max.unsqueeze(1)

    weights_sum = torch.sum(box_weights_expanded, dim=2, keepdim=True) + eps # (B, M, 1, 1)

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

    intersection_box_dists = BoxDistribution(intersection_box_mu_min, intersection_box_mu_max, intersection_box_beta_min, intersection_box_beta_max)

    return intersection_box_dists


def pairwise_gumbel_intersection(box_dists_A: BoxDistribution, box_dists_B: BoxDistribution, eps: float = EPS) -> BoxDistribution:
    """
    Computes the intersection of every box in A with every box in B.
    
    Args:
        box_dists_A: Shape (N, D)
        box_dists_B: Shape (M, D)

    Formula:
        Let A and B be the two sets of input box distributions.
        The intersection parameters (mu_new, beta_new) for a pair (A_i, B_j) are computed as:

        1. Intersection Uncertainty (Arithmetic Mean Approximation):
           beta_new = (beta_A + beta_B) / 2

        2. Intersection Min Coordinate (Smooth Maximum):
           mu_min_new = beta_new * log(exp(mu_min_A / beta_new) + exp(mu_min_B / beta_new))
                      = beta_new * logaddexp(mu_min_A / beta_new, mu_min_B / beta_new)

        3. Intersection Max Coordinate (Smooth Minimum):
           mu_max_new = -beta_new * log(exp(-mu_max_A / beta_new) + exp(-mu_max_B / beta_new))
                      = -beta_new * logaddexp(-mu_max_A / beta_new, -mu_max_B / beta_new)
        
    Returns:
        intersection_box_dists: Shape (N, M, D)
    """
    
    # Broadcast Shapes
    
    # Box A: (N, 1, D)
    mu_min_a = box_dists_A.mu_min.unsqueeze(1)
    mu_max_a = box_dists_A.mu_max.unsqueeze(1)
    beta_min_a = box_dists_A.beta_min.unsqueeze(1)
    beta_max_a = box_dists_A.beta_max.unsqueeze(1)

    # Box B: (1, M, D)
    mu_min_b = box_dists_B.mu_min.unsqueeze(0)
    mu_max_b = box_dists_B.mu_max.unsqueeze(0)
    beta_min_b = box_dists_B.beta_min.unsqueeze(0)
    beta_max_b = box_dists_B.beta_max.unsqueeze(0)

    
    # Compute Intersection Parameters
    
    # Beta: We approximate the intersection beta as the mean of the two betas
    new_beta_min = (beta_min_a + beta_min_b) / 2.0 # (N, M, D)
    new_beta_max = (beta_max_a + beta_max_b) / 2.0 # (N, M, D)

    # Mu Min: Smooth Maximum (LogSumExp(x)) (beta * log(exp(mu_a/beta) + exp(mu_b/beta)))
    arg_min_a = mu_min_a / (new_beta_min + eps) # (N, M, D)
    arg_min_b = mu_min_b / (new_beta_min + eps) # (N, M, D)
    new_mu_min = new_beta_min * torch.logaddexp(arg_min_a, arg_min_b) # (N, M, D)

    # Mu Max: Smooth Minimum (-LogSumExp(-x)) (-beta * log(exp(-mu_a/beta) + exp(-mu_b/beta)))
    arg_max_a = -mu_max_a / (new_beta_max + eps) # (N, M, D)
    arg_max_b = -mu_max_b / (new_beta_max + eps) # (N, M, D)
    new_mu_max = -new_beta_max * torch.logaddexp(arg_max_a, arg_max_b) # (N, M, D)

    return BoxDistribution(new_mu_min, new_mu_max, new_beta_min, new_beta_max)


def gumbel_sigmoid(logits, temperature=1.0, hard=False, eps=EPS):
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
    
    # Gumbel-Sigmoid sampling (sigmoid((logits + gumbel_noise) / temp))
    y_soft = torch.sigmoid((logits + gumbels) / temperature)
    
    if hard:
        # Straight-Through Estimator
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
        
        return ret
    
    else:
        return y_soft