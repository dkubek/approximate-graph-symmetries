"""
Implements a Dimensionality Reduction method for the Approximate Symmetry Problem.

This module provides a solver for the Approximate Symmetry Problem (ASP) based
on the SoftSort operator. This approach fundamentally changes the optimization
by reducing the dimensionality of the search space from O(n^2) for a matrix
to O(n) for a vector.

The core idea is that a permutation can be implicitly defined by the sorting
order of a vector of n scores. The SoftSort operator provides a differentiable
relaxation of the `argsort` operation, which maps a score vector to a
doubly stochastic matrix. As a temperature parameter `tau` approaches zero, this
matrix converges to the permutation matrix corresponding to the sort order.

This parameterization transforms the highly constrained problem of finding a
permutation matrix into an unconstrained optimization problem over an n-dimensional
vector, which can be solved efficiently with standard gradient-based optimizers.

Reference:
    Prillo, S., & Eisenschlos, J. (2020). "Softsort: A continuous relaxation
    for the argsort operator". International Conference on Machine Learning.
"""
from time import time

import numpy as np
import torch
from tqdm import tqdm

from AGS.annealing import get_annealing_tau


def soft_sort(s, tau):
    """
    PyTorch implementation of SoftSort function for 1D tensors.

    Args:
        s: Input tensor of shape (1,)
        tau: Temperature parameter for SoftSort

    Returns:
        P_hat: Soft permutation matrix of shape (n, n)
    """

    # Check that input is 1D
    if s.dim() != 1:
        raise ValueError(
            f"Expected 1D tensor of shape [sequence_length,], got tensor with {s.dim()} dimensions"
        )

    # Sort the tensor in descending order
    s_sorted, _ = torch.sort(s, dim=0, descending=True)

    # Reshape tensors for computing pairwise distances
    # Convert s to shape [sequence_length, 1] and s_sorted to shape [1, sequence_length]
    s_col = s.unsqueeze(1)  # Shape: [sequence_length, 1]
    s_sorted_row = s_sorted.unsqueeze(0)  # Shape: [1, sequence_length]

    # Compute pairwise distances
    # Broadcasting will create a tensor of shape [sequence_length, sequence_length]
    pairwise_distances = -torch.abs(s_col - s_sorted_row)

    # Apply softmax along the last dimension (dim=1 for a 2D tensor)
    P_hat = torch.nn.functional.softmax(pairwise_distances / tau, dim=1)

    return P_hat


class DimensionalityReduction:
    """
    Solves the Approximate Symmetry Problem using the SoftSort operator.

    This class implements an optimization procedure that finds an approximate
    symmetry by optimizing a low-dimensional score vector `s`. The `soft_sort`
    function is used to differentiably map `s` to a doubly stochastic matrix,
    which is then used to evaluate the ASP objective function. The optimization
    is performed using AdamW, and an annealing schedule on the temperature `tau`
    guides the solution towards a discrete permutation.
    """
    def __init__(
        self,
        max_iter=2000,
        loss="indefinite",
        initial_tau=1,
        final_tau=1e-6,
        annealing_scheme="cosine",
        decay_steps=None,
        learning_rate=1e-1,
        min_rel_improvement=1e-4,
        verbose=1,
    ):
        """
        Initializes the DimensionalityReduction solver.

        Args:
            max_iter (int): Maximum number of optimization iterations.
            loss (str): The loss function to use. Either "indefinite" for the
                trace-based formulation or "convex" for the Frobenius norm
                formulation.
            initial_tau (float): The starting temperature for the SoftSort
                operator. Higher values produce "softer" matrices.
            final_tau (float): The final temperature for annealing. As tau
                approaches zero, the output approaches a hard permutation matrix.
            annealing_scheme (str): The annealing schedule for `tau`. Options
                are "cosine", "linear", or "exponential".
            decay_steps (int): The number of iterations over which `tau`
                anneals from its initial to final value.
            learning_rate (float): The learning rate for the AdamW optimizer.
            min_rel_improvement (float): The minimum relative improvement in
                loss required to update the best solution found so far.
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
        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        if self.verbose:
            print(f"Using {self.device} device")

    def _loss_trace(self, A, P, c):
        quadratic_term = -torch.trace(A @ P @ A.transpose(-2, -1) @ P.transpose(-2, -1))

        diag_elements = torch.diagonal(P, dim1=-2, dim2=-1)

        # Classic penalty
        penalty = torch.sum(c * diag_elements)

        return quadratic_term + penalty

    def _loss_norm(self, A, P, c):
        quadratic_term = (
            1 / 2 * torch.sum(torch.pow(P @ A @ P.transpose(-2, -1) - A, 2))
        )

        diag_elements = torch.diagonal(P, dim1=-2, dim2=-1)

        # Classic penalty
        penalty = torch.sum(c * diag_elements)

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
                - "P" (np.ndarray): The final n x n permutation matrix.
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

        # Initialize s on the unit sphere
        s = torch.nn.Parameter(
            torch.nn.functional.normalize(
                torch.rand(n, device=self.device, dtype=self.dtype), dim=0
            ),
            requires_grad=True,
        )
        optimizer = torch.optim.AdamW([s], lr=self.learning_rate)

        # Initialize tracking variables
        best_loss = float("inf")
        best_s = s
        best_iteration = 0

        pbar = tqdm(range(self.max_iter), disable=(self.verbose < 1))
        start_time = time()
        iterations_performed = 0

        for iteration in pbar:
            iterations_performed = iteration + 1

            # Compute current tau using the annealing function
            current_tau = get_annealing_tau(
                iteration,
                self.decay_steps,
                self.initial_tau,
                self.final_tau,
                scheme=self.annealing_scheme,
            )

            # Training step
            optimizer.zero_grad()
            perm_matrix = soft_sort(s, tau=current_tau)
            loss_train = self.loss(A, perm_matrix, c=c)
            loss_train.backward()
            optimizer.step()

            # Compute validation loss
            with torch.no_grad():
                perm_matrix_val = soft_sort(s, tau=1e-8)
                loss_val = self.loss(A, perm_matrix_val, c=0)

            # Check for improvement
            if loss_val < best_loss * (1 - self.min_rel_improvement):
                best_loss = loss_val
                best_s = s.clone().detach()
                best_iteration = iteration + 1

            # Update progress bar
            pbar.set_description(f"Loss: {loss_val.item():.6f}, Tau: {current_tau:.4f}")

        end_time = time()

        P_opt = soft_sort(best_s, tau=1e-8).cpu().numpy()

        return {
            "P": P_opt,
            "metrics": {
                "time": end_time - start_time,
                "iterations": iterations_performed,
                "best_iteration": best_iteration,
            },
        }
