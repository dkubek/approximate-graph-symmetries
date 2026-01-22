"""
SA: Symmetric Approximator with Frank-Wolfe and automatic differentiation.

This module implements a Frank-Wolfe solver for the Relaxed Approximate
Symmetry Problem (rASP) with support for arbitrary differentiable objective
functions. Gradients are computed via PyTorch autograd, and step sizes are
determined using Armijo backtracking line search.

The Frank-Wolfe algorithm operates on the Birkhoff polytope. At each iteration:
1. Compute gradient of objective via autograd
2. Solve linear assignment problem to find descent direction
3. Find step size via Armijo line search
4. Update P as convex combination

References:
    - Frank, M., & Wolfe, P. (1956). "An algorithm for quadratic programming."
    - Armijo, L. (1966). "Minimization of functions having Lipschitz continuous
      first partial derivatives."
"""

import time
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from AGS.initialization import (
    check_random_state,
    init_barycenter,
    init_random_doubly_stochastic,
    init_random_permutation,
)
from AGS.methods.sa.objectives import get_objective


class SymmetricApproximator:
    """
    Frank-Wolfe solver for rASP with automatic differentiation.

    Supports arbitrary differentiable objective functions. Uses Armijo
    backtracking line search for step size selection.
    """

    def __init__(
        self,
        objective: Union[str, Callable] = "nonconvex_relaxation",
        max_iter: int = 500,
        tol: float = 1e-8,
        armijo_beta: float = 0.5,
        armijo_sigma: float = 1e-4,
        rng: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        verbose: int = 1,
    ):
        """
        Initialize the Symmetric Approximator solver.

        Args:
            objective: Objective function. Either a string name from
                {'nonconvex_relaxation', 'dynamic_fixed_points', 'frobenius_squared'}
                or a callable with signature f(A, P, c) -> scalar.
            max_iter: Maximum Frank-Wolfe iterations.
            tol: Convergence tolerance for change in P (Frobenius norm).
            armijo_beta: Step size reduction factor for backtracking (0 < beta < 1).
            armijo_sigma: Sufficient decrease parameter (0 < sigma < 1).
            rng: Random seed or generator for initialization.
            verbose: Verbosity level. 0=silent, 1=progress bar.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.armijo_beta = armijo_beta
        self.armijo_sigma = armijo_sigma
        self.rng = check_random_state(rng)
        self.verbose = verbose

        # Resolve objective function
        if isinstance(objective, str):
            self.objective = get_objective(objective)
            self.objective_name = objective
        else:
            self.objective = objective
            self.objective_name = getattr(objective, "__name__", "custom")

    def _compute_gradient(
        self,
        A: torch.Tensor,
        P: torch.Tensor,
        c: torch.Tensor,
    ) -> np.ndarray:
        """Compute gradient of objective w.r.t. P using autograd."""
        P_var = P.clone().detach().requires_grad_(True)
        loss = self.objective(A, P_var, c)
        loss.backward()
        return P_var.grad.detach().numpy()

    def _armijo_line_search(
        self,
        A: torch.Tensor,
        P: np.ndarray,
        Q: np.ndarray,
        c: torch.Tensor,
        grad: np.ndarray,
    ) -> float:
        """
        Armijo backtracking line search.

        Finds step size alpha such that:
            f(P + alpha*d) <= f(P) + sigma * alpha * grad^T @ d

        where d = Q - P is the descent direction.

        Args:
            A: Adjacency matrix (torch tensor)
            P: Current iterate (numpy array)
            Q: Frank-Wolfe vertex (numpy array)
            c: Penalty vector (torch tensor)
            grad: Gradient at P (numpy array)

        Returns:
            Step size alpha in (0, 1]
        """
        d = Q - P
        directional_deriv = np.sum(grad * d)

        # If not a descent direction, return 0
        if directional_deriv >= 0:
            return 0.0

        P_torch = torch.from_numpy(P)
        f_P = self.objective(A, P_torch, c).item()

        alpha = 1.0
        max_backtracks = 20

        for _ in range(max_backtracks):
            P_new = P + alpha * d
            P_new_torch = torch.from_numpy(P_new)
            f_new = self.objective(A, P_new_torch, c).item()

            # Armijo condition: sufficient decrease
            if f_new <= f_P + self.armijo_sigma * alpha * directional_deriv:
                return alpha

            alpha *= self.armijo_beta

        return alpha

    def solve(
        self,
        A: np.ndarray,
        c: Union[float, np.ndarray],
        P0: Union[
            Literal["barycenter", "random_permutation", "random_doubly_stochastic"],
            np.ndarray,
        ] = "random_doubly_stochastic",
    ) -> dict:
        """
        Execute Frank-Wolfe algorithm to solve rASP.

        Args:
            A: Adjacency matrix (n, n)
            c: Penalty parameter. Scalar or vector of length n.
            P0: Initial doubly stochastic matrix or initialization method.

        Returns:
            dict with keys:
                - 'P': Final doubly stochastic matrix (n, n)
                - 'metrics': dict with 'time', 'iterations', 'final_loss'
        """
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError("A must be a square matrix")

        # Handle c parameter
        if np.isscalar(c):
            c_np = np.full(n, c, dtype=np.float64)
        else:
            c_np = np.asarray(c, dtype=np.float64)
            if c_np.shape != (n,):
                raise ValueError(f"c must be a scalar or vector of length {n}")

        # Initialize P
        if isinstance(P0, np.ndarray):
            P = P0.astype(np.float64).copy()
            if (
                P.shape != (n, n)
                or np.abs(P.sum(axis=1) - 1).max() > 1e-6
                or np.abs(P.sum(axis=0) - 1).max() > 1e-6
                or (P < 0).any()
            ):
                raise ValueError("P0 must be a doubly stochastic matrix")
        elif P0 == "barycenter":
            P = init_barycenter(n)
        elif P0 == "random_permutation":
            P = init_random_permutation(n, self.rng)
        elif P0 == "random_doubly_stochastic":
            P = init_random_doubly_stochastic(n, self.rng)
        else:
            raise ValueError(f"Unknown initialization method: {P0}")

        # Convert to torch tensors (for objective evaluation)
        A_torch = torch.from_numpy(A.astype(np.float64))
        c_torch = torch.from_numpy(c_np)

        # Frank-Wolfe iterations
        pbar = tqdm(range(self.max_iter), disable=(self.verbose < 1))
        start_time = time.time()
        iterations_performed = 0

        for iteration in pbar:
            iterations_performed = iteration + 1
            P_prev = P.copy()

            # Step 1: Compute gradient via autograd
            P_torch = torch.from_numpy(P)
            grad = self._compute_gradient(A_torch, P_torch, c_torch)

            # Step 2: Solve linear subproblem (LAP)
            _, col_ind = linear_sum_assignment(grad)
            Q = np.eye(n)[col_ind]

            # Step 3: Armijo line search for step size
            alpha = self._armijo_line_search(A_torch, P, Q, c_torch, grad)

            # Step 4: Update P
            P = P + alpha * (Q - P)

            # Check convergence
            change = np.linalg.norm(P - P_prev, "fro") / np.sqrt(n)
            if change < self.tol:
                break

            pbar.set_description(f"alpha={alpha:.3f}, change={change:.2e}")

        solve_time = time.time() - start_time

        # Compute final loss
        P_torch = torch.from_numpy(P)
        final_loss = self.objective(A_torch, P_torch, c_torch).item()

        return {
            "P": P,
            "metrics": {
                "time": solve_time,
                "iterations": iterations_performed,
                "final_loss": final_loss,
                "objective": self.objective_name,
            },
        }
