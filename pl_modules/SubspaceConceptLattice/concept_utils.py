import torch
import torch.nn.functional as F

def get_trace(P):
    """Computes the trace (effective rank/dimension) of batched projectors."""
    return torch.einsum("bii->b", P)

def get_similarity(P1, P2):
    """Computes the unnormalized similarity (overlap) between two subspaces."""
    prod = torch.bmm(P1, P2.transpose(1, 2))
    return get_trace(prod)

def get_inclusion(P_sub, P_super, eps=1e-6):
    """
    Computes P(P_super | P_sub) : The degree to which P_sub is contained in P_super.
    Returns a score from 0 to 1.
    """
    overlap = get_similarity(P_sub, P_super)
    prior = get_trace(P_sub)
    return torch.clamp(overlap / (prior + eps), eps, 1.0 - eps)

def get_binary_inclusion(A, B):
    """
    Computes asymmetric geometric penalty that only activates if A_i strictly exceeds B_i.
    A is subset, B is superset.
    Returns 1 - mean(max(0, A_i - B_i))
    """
    A_flat = A.reshape(A.shape[0], -1)
    B_flat = B.reshape(B.shape[0], -1)
    return 1.0 - torch.mean(F.relu(A_flat - B_flat), dim=1)

def ridge_projector(x: torch.Tensor, lbd: float = 0.05) -> torch.Tensor:
    """
    Computes the smooth projection operator using Tikhonov regularization.
    Formula: X (X^T X + \lambda I)^-1 X^T
    """
    B, N, D = x.shape
    # gram matrix: X X^T
    gram = torch.bmm(x, x.transpose(1, 2)) 
    # Add ridge penalty \lambda I
    gram_ridge = gram + lbd * torch.eye(N, device=x.device).expand(B, N, N)
    
    # Solve the linear system to get the projector
    proj = torch.bmm(x.transpose(1, 2), torch.linalg.solve(gram_ridge, x))
    return proj

def gumbel_sigmoid(logits, tau=0.25, hard=False, eps=1e-10):
    # Sample from Gumbel(0, 1)
    U = torch.rand_like(logits)
    U = torch.clamp(U, eps, 1.0 - eps) # CRITICAL: Prevents log(0)
    gumbel_noise = -torch.log(-torch.log(U))
    # Add noise and apply sigmoid with temperature
    y_soft = torch.sigmoid((logits + gumbel_noise) / tau)
    
    if hard:
        # Straight-Through Estimator (STE)
        # Forward pass is exactly 0 or 1. Backward pass uses the gradient of y_soft.
        y_hard = (y_soft > 0.5).float()
        y = y_hard - y_soft.detach() + y_soft
        return y
    else:
        return y_soft