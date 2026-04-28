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

def get_residual_inclusion(P_sub, P_super, gamma=1.0, eps=1e-6):
    """
    Computes inclusion of P_sub in P_super without division by Tr(P_sub).

    Measures the residual volume of P_sub that falls outside P_super:
        error = ReLU(Tr(P_sub) - Tr(P_sub @ P_super))
    Then maps to [0, 1] via a decaying exponential:
        score = exp(-gamma * error)

    Key property: when P_sub = 0 (empty set / universal concept intent),
    error = 0 and score = 1.0. The empty set is perfectly included in
    everything — a lattice axiom that the trace-ratio formula violates.

    Args:
        P_sub:   (B, D, D) projector for the candidate subset
        P_super: (B, D, D) projector for the candidate superset
        gamma:   controls strictness. Higher = sharper penalty for residual.
                 Scale as ~1/n_attr if ambient_dim grows.
        eps:     clamp floor for BCE compatibility

    Returns:
        (B,) scores in [eps, 1 - eps]
    """
    trace_sub = get_trace(P_sub)                        # (B,)
    overlap = get_similarity(P_sub, P_super)             # (B,)
    residual_error = F.relu(trace_sub - overlap)         # (B,)
    score = torch.exp(-gamma * residual_error)           # (B,)
    return torch.clamp(score, eps, 1.0 - eps)


def get_binary_inclusion(A, B):
    """
    Computes asymmetric geometric penalty that only activates if A_i strictly exceeds B_i.
    A is subset, B is superset.
    
    Returns 1 - mean(max(0, A_i - B_i))
    """
    A_flat = A.reshape(A.shape[0], -1)
    B_flat = B.reshape(B.shape[0], -1)
    return 1.0 - torch.mean(F.relu(A_flat - B_flat), dim=1)

class _StableRidgeProjectorFn(torch.autograd.Function):
    """Ridge projector with analytically computed, numerically stable gradients.

    Forward:  P = X^T A^{-1} X  where A = XX^T + λI,  computed via linalg.solve.
    Backward: analytical gradient dL/dX = 2 G Ω_s (I - P), where G = A^{-1}X
              and Ω_s = (dL/dP + dL/dP^T) / 2.

    This avoids backpropagating through linalg.solve, SVD, or eigh — all of which
    produce NaN on degenerate/ill-conditioned inputs. Verified via torch.autograd.gradcheck.
    """

    @staticmethod
    def forward(ctx, x, lbd):
        B, N, D = x.shape
        gram = torch.bmm(x, x.transpose(1, 2))  # (B, N, N)
        gram_ridge = gram + lbd * torch.eye(N, device=x.device, dtype=x.dtype).expand(B, N, N)
        # G = A^{-1} X where A = XX^T + λI
        G = torch.linalg.solve(gram_ridge, x)  # (B, N, D)
        # P = X^T G
        P = torch.bmm(x.transpose(1, 2), G)  # (B, D, D)
        ctx.save_for_backward(x, G, P)
        ctx.lbd = lbd
        return P

    @staticmethod
    def backward(ctx, grad_P):
        x, G, P = ctx.saved_tensors
        B, N, D = x.shape
        # grad_P: (B, D, D) — dL/dP = Ω
        # Symmetrize since P is symmetric
        Omega_s = (grad_P + grad_P.transpose(1, 2)) / 2.0  # (B, D, D)

        # Analytical gradient: dL/dX = 2 G Ω_s (I - P)
        I_minus_P = torch.eye(D, device=x.device, dtype=x.dtype).expand(B, D, D) - P
        grad_x = 2.0 * torch.bmm(G, torch.bmm(Omega_s, I_minus_P))  # (B, N, D)

        return grad_x, None


def ridge_projector(x: torch.Tensor, lbd: float = 0.05) -> torch.Tensor:
    """
    Computes the smooth projection operator using Tikhonov regularization.
    Formula: P = X^T (X X^T + λI)^{-1} X

    Uses linalg.solve for the forward pass and an analytical gradient formula
    for the backward pass. The gradient uses only the pre-computed
    G = (XX^T + λI)^{-1} X and P = X^T G, requiring no matrix decomposition
    backward (which is the source of NaN in degenerate cases).

    Gradient formula: dL/dX = 2 G Ω_s (I - P)
    where Ω_s = (dL/dP + dL/dP^T) / 2 is the symmetrized output gradient.
    """
    return _StableRidgeProjectorFn.apply(x, lbd)

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