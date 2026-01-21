"""
Implements the Orthogonal Relaxation (OR) method for the Approximate Symmetry Problem.

This module provides a solver for the Approximate Symmetry Problem (ASP) by
relaxing the discrete set of permutation matrices to the non-convex manifold of
orthogonal matrices. This approach offers a significant reduction in the number
of optimization parameters compared to methods operating on the Birkhoff polytope.

The core of this method is the OT4P (Orthogonal Group-based Transformation for
Permutation Relaxation) framework, which provides a differentiable mapping from
an unconstrained vector space to the set of permutation matrices. The optimization
is performed using gradient-based methods (AdamW) with a temperature parameter
that controls the interpolation between the orthogonal group and the permutation
matrices, guided by an annealing schedule.

The module supports both the indefinite and convex formulations of the relaxed ASP
objective function.

Reference:
    Guo, Y., Zhu, H., Wu, T., et al. (2024). "OT4P: Unlocking Effective
    Orthogonal Group Path for Permutation Relaxation". Advances in Neural
    Information Processing Systems.
"""

from time import time

import numpy as np
import torch
from tqdm import tqdm

from AGS.annealing import get_annealing_tau

from .OT4P.ot4p import OT4P


class OrthogonalRelaxation:
    """
    Solves the Approximate Symmetry Problem using Orthogonal Relaxation.

    This class implements an optimization procedure for the ASP by relaxing the
    search space from permutation matrices to the orthogonal group. It utilizes
    the OT4P framework to parameterize the problem, enabling unconstrained
    optimization with gradient-based methods in PyTorch. The method iteratively
    refines a solution while an annealing schedule guides the relaxation towards
    a discrete permutation matrix.
    """

    def __init__(
        self,
        max_iter=3000,
        initial_tau=0.7,
        final_tau=0.5,
        loss="indefinite",
        annealing_scheme="exponential",
        decay_steps=5000,
        learning_rate=1e-1,
        min_rel_improvement=1e-4,
        verbose=1,
    ):
        """
        Initializes the OrthogonalRelaxation solver.

        Args:
            max_iter (int): Maximum number of optimization iterations.
            initial_tau (float): The starting temperature for annealing. Higher
                values correspond to a weaker approximation of permutations.
            final_tau (float): The final temperature for annealing.
            loss (str): The loss function to use. Either "indefinite" for the
                trace-based formulation or "convex" for the Frobenius norm
                formulation.
            annealing_scheme (str): The annealing schedule for the temperature
                tau. Options are "cosine", "linear", or "exponential".
            decay_steps (int): The number of iterations over which tau anneals
                from its initial to final value.
            learning_rate (float): The learning rate for the AdamW optimizer.
            min_rel_improvement (float): The minimum relative improvement in
                loss required to update the best solution found so far. Used
                for early stopping.
            verbose (int): Verbosity level. 0 for silent, 1 for progress bar.
        """

        self.max_iter = max_iter
        self.learning_rate = learning_rate

        if loss == "indefinite":
            self.loss = self._loss_trace
        elif loss == "convex":
            self.loss = self._loss_norm
        else:
            ValueError(f"Loss {loss} is not recognized!")

        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.annealing_scheme = annealing_scheme
        self.decay_steps = decay_steps or max_iter

        self.min_rel_improvement = min_rel_improvement

        self.verbose = verbose

        self.dtype = torch.float
        self.device = "cpu"
        if self.verbose:
            print(f"Using {self.device} device")

    def _loss_trace(self, A, P, c):
        quadratic_term = -torch.trace(A @ P @ A.transpose(-2, -1) @ P.transpose(-2, -1))

        diag_elements = torch.diagonal(P, dim1=-2, dim2=-1)

        # Classic penalty
        positive_diags = torch.pow(diag_elements, 2)
        penalty = torch.sum(c * positive_diags)

        return quadratic_term + penalty

    def _loss_norm(self, A, P, c):
        quadratic_term = (
            1 / 2 * torch.sum(torch.pow(P @ A @ P.transpose(-2, -1) - A, 2))
        )

        diag_elements = torch.diagonal(P, dim1=-2, dim2=-1)

        # Classic penalty
        positive_diags = torch.pow(diag_elements, 2)
        penalty = torch.sum(c * positive_diags)

        return quadratic_term + penalty

    def solve(
        self,
        A,
        c=0.2,
    ):
        """
        Executes the optimization to find an approximate symmetry.

        Args:
            A (np.ndarray): The n x n adjacency matrix of the graph.
            c (Union[float, np.ndarray]): The penalty parameter for the diagonal
                (fixed points). Can be a scalar or a vector of length n.

        Returns:
            dict: A dictionary containing:
                - "P" (np.ndarray): The optimized n x n permutation matrix.
                - "metrics" (dict): A dictionary with performance metrics,
                  including 'time', 'iterations', and 'best_iteration'.
        """

        n = A.shape[0]

        # Handle scalar c
        if np.isscalar(c):
            c = np.full(n, c)
        else:
            c = np.asarray(c)

        # Validate inputs
        assert A.shape == (n, n), "A must be square"
        assert c.shape == (n,), f"c must have shape ({n},), got {c.shape}"

        # Convert to PyTorch tensors
        A = torch.from_numpy(A.astype(float)).to(device=self.device, dtype=self.dtype)
        c = torch.from_numpy(c.astype(float)).to(device=self.device, dtype=self.dtype)

        model = OT4P(n).to(device=self.device, dtype=self.dtype)
        weightP = torch.nn.Parameter(
            torch.rand_like(A, device=self.device, dtype=self.dtype), requires_grad=True
        )
        optimizer = torch.optim.AdamW([weightP], lr=self.learning_rate)

        # Initialize tracking variables
        best_loss = float("inf")
        best_weight = weightP.clone().detach()
        best_iteration = 0

        pbar = tqdm(range(self.max_iter), disable=(self.verbose < 1))
        start_time = time()
        iterations_performed = 0

        for i in pbar:
            iterations_performed = i + 1
            current_tau = get_annealing_tau(
                i,
                self.decay_steps,
                self.initial_tau,
                self.final_tau,
                scheme=self.annealing_scheme,
            )

            # Training step
            optimizer.zero_grad()
            perm_matrix = model(weightP, tau=current_tau)
            loss_val = self.loss(A, perm_matrix, c)
            loss_val.backward()
            optimizer.step()

            # Compute validation loss
            with torch.no_grad():
                perm_matrix_val = model(weightP, tau=0)
                loss_val = self.loss(A, perm_matrix_val, c)

            # Check for improvement
            if loss_val < best_loss * (1 - self.min_rel_improvement):
                best_loss = loss_val
                best_weight = weightP.clone().detach()
                best_iteration = i + 1

            # Update model base
            model.update_base(weightP)

            # Update progress bar
            pbar.set_description(f"Loss: {loss_val.item():.6f}, Tau: {current_tau:.4f}")

        end_time = time()

        P_opt = model(best_weight, tau=0).cpu().numpy()

        return {
            "P": P_opt,
            "metrics": {
                "time": end_time - start_time,
                "iterations": iterations_performed,
                "best_iteration": best_iteration,
            },
        }
