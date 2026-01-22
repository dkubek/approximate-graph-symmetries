"""
Objective functions for the Symmetric Approximator (SA) solver.

Each objective is a callable that takes (A, P, c) and returns a scalar loss.
The functions are written in PyTorch to enable automatic differentiation.

Objectives:
    - nonconvex_relaxation: -tr(A @ P @ A.T @ P.T)
    - dynamic_fixed_points: E / (n(n-1) - 2F(F-1)) where F = tr(P)
    - frobenius_squared: ||A - P @ A @ P.T||_F^2 + c * tr(P)
"""

import torch


def nonconvex_relaxation(
    A: torch.Tensor, P: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """
    Non-convex relaxation objective: -tr(A @ P @ A.T @ P.T) + c * tr(P).

    This maximizes the alignment between A and P @ A @ P.T by minimizing
    the negative trace of their product.

    Args:
        A: Adjacency matrix (n, n)
        P: Doubly-stochastic matrix (n, n)
        c: Penalty vector for diagonal (n,)

    Returns:
        Scalar loss value
    """
    # -tr(A @ P @ A.T @ P.T) = -sum((A @ P @ A.T) * P.T)
    term1 = -torch.sum((A @ P @ A.T) * P.T)
    term2 = torch.sum(c * torch.diag(P))
    return term1 + term2


def dynamic_fixed_points(
    A: torch.Tensor, P: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """
    Dynamic fixed points objective: E / (n(n-1) - 2F(F-1)).

    Normalizes the Frobenius error by the "effective" number of non-fixed entries,
    where F = tr(P) is a relaxed count of fixed points.

    Note: The penalty c is not used here since fixed points are already
    penalized through the denominator structure.

    Args:
        A: Adjacency matrix (n, n)
        P: Doubly-stochastic matrix (n, n)
        c: Penalty vector (unused, kept for interface consistency)

    Returns:
        Scalar loss value
    """
    n = A.shape[0]
    E = torch.norm(A - P @ A @ P.T, p="fro")
    F = torch.trace(P)

    # Denominator: effective non-fixed pairs
    denominator = n * (n - 1) - 2 * F * (F - 1) + 1e-8

    return E / denominator


def frobenius_squared(
    A: torch.Tensor, P: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """
    Frobenius squared objective: ||A - P @ A @ P.T||_F^2 + c * tr(P).

    This is the standard ASP objective with a penalty term on fixed points.

    Args:
        A: Adjacency matrix (n, n)
        P: Doubly-stochastic matrix (n, n)
        c: Penalty vector for diagonal (n,)

    Returns:
        Scalar loss value
    """
    diff = A - P @ A @ P.T
    term1 = torch.sum(diff * diff)
    term2 = torch.sum(c * torch.diag(P))
    return term1 + term2


# Registry mapping string names to objective functions
OBJECTIVES = {
    "nonconvex_relaxation": nonconvex_relaxation,
    "dynamic_fixed_points": dynamic_fixed_points,
    "frobenius_squared": frobenius_squared,
}


def get_objective(name: str):
    """
    Retrieve an objective function by name.

    Args:
        name: One of 'nonconvex_relaxation', 'dynamic_fixed_points', 'frobenius_squared'

    Returns:
        Callable objective function

    Raises:
        ValueError: If name is not recognized
    """
    if name not in OBJECTIVES:
        available = ", ".join(OBJECTIVES.keys())
        raise ValueError(f"Unknown objective '{name}'. Available: {available}")
    return OBJECTIVES[name]
