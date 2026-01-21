"""
Implements the Manifold Optimization method for the Approximate Symmetry Problem.

This module provides a solver for the Relaxed Approximate Symmetry Problem (rASP)
by framing it as an unconstrained optimization problem on a Riemannian manifold.
Instead of treating the constraints of the Birkhoff polytope algebraically, this
approach embraces the geometry of the set of doubly stochastic matrices.

The interior of the Birkhoff polytope is treated as the Doubly Stochastic
Manifold, equipped with the Fisher Information Metric. This allows the use of
standard gradient-based optimization algorithms, such as Steepest Descent or
Trust Regions, adapted to operate on this curved space. Key operations like
gradient computation and movement (retraction) are defined by the manifold's
geometry.

This implementation uses the `pymanopt` library, which automates many of the
complex manifold operations and integrates with PyTorch for automatic
differentiation of the objective function.

References:
    - Boumal, N. (2023). "An introduction to optimization on smooth manifolds."
    - Douik, A., & Hassibi, B. (2019). "Manifold optimization over the set of
      doubly stochastic matrices: A second-order geometry."
"""

import time
from typing import Optional

import numpy as np
import pymanopt
import torch
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions

from AGS.methods.manifold.doublystochastic import DoublyStochastic


class Manifold:
    """
    Solves the rASP using optimization on the Doubly Stochastic Manifold.

    This class provides an interface to the `pymanopt` library to solve the
    rASP. It sets up the problem by defining the cost function and the
    Doubly Stochastic manifold, then runs a chosen Riemannian optimizer to
    find the optimal doubly stochastic matrix.
    """

    def __init__(
        self,
        optimizer="steepest_descent",
        max_iter=500,
        verbose=1,
        **optimizer_kwargs,
    ):
        """
        Initializes the Manifold solver.

        Args:
            optimizer (str): The Riemannian optimizer to use. Supported options
                are "steepest_descent", "trust_regions", and
                "conjugate_gradient".
            max_iter (int): The maximum number of iterations for the optimizer.
            verbose (int): Verbosity level for the `pymanopt` solver.
                0 for quiet, 1 for summary, 2 for detailed iteration output.
            **optimizer_kwargs: Additional keyword arguments to be passed
                directly to the chosen `pymanopt` optimizer.
        """

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.optimizer_kwargs = optimizer_kwargs

        self.verbose = verbose

        self.dtype = torch.float64
        self.device = "cpu"
        if self.verbose > 0:
            print(f"Using {self.device} device")

    def solve(
        self,
        A,
        c,
        P0: Optional[np.ndarray] = None,
    ):
        """
        Executes the manifold optimization to solve the rASP.

        Args:
            A (np.ndarray): The n x n adjacency matrix of the graph.
            c (Union[float, np.ndarray]): The penalty parameter for the diagonal
                (fixed points). Can be a scalar or a vector of length n.
            P0 (np.ndarray, optional): An initial doubly stochastic matrix to
                start the optimization. If None, the optimizer's default
                initialization is used.

        Returns:
            dict: A dictionary containing:
                - "P" (np.ndarray): The optimized doubly stochastic matrix.
                - "metrics" (dict): A dictionary with performance metrics,
                  including 'time' and 'iterations'.
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

        # Create manifold
        manifold = DoublyStochastic(n, retraction_method="sinkhorn")

        # Define cost function
        @pymanopt.function.pytorch(manifold)
        def cost(P):
            """Regularized cost with entropy and annealing."""
            # Original objective
            AP = A @ P
            APAT = AP @ A.t()
            term1 = -torch.sum(APAT * P)
            # term1 = torch.mean(torch.pow(P @ A @ P.t() - A, 2))
            term2 = torch.sum(c * torch.diag(P))

            total_cost = term1 + term2

            return total_cost

        # Create problem
        problem = Problem(manifold, cost)

        # Choose and configure optimizer
        if self.optimizer == "trust_regions":
            opt = TrustRegions(
                max_iterations=self.max_iter,
                verbosity=self.verbose,
                **self.optimizer_kwargs,
            )
        elif self.optimizer == "steepest_descent":
            opt = SteepestDescent(
                max_iterations=self.max_iter,
                verbosity=self.verbose,
                **self.optimizer_kwargs,
            )
        elif self.optimizer == "conjugate_gradient":
            opt = ConjugateGradient(
                max_iterations=self.max_iter,
                verbosity=self.verbose,
                **self.optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Run optimization
        start_time = time.time()
        result = opt.run(problem, initial_point=P0)
        end_time = time.time()

        # Convert the result back to numpy
        P_opt = result.point

        return {
            "P": P_opt,
            "metrics": {"time": end_time - start_time, "iterations": result.iterations},
        }
