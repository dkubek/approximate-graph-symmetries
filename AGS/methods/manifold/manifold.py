import time
from typing import Optional

import numpy as np
import pymanopt
import torch
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions

from AGS.methods.manifold.doublystochastic import DoublyStochastic


class Manifold:
    def __init__(
        self,
        optimizer="steepest_descent",
        max_iter=500,
        verbose=1,
        **optimizer_kwargs,
    ):
        """
        optimizer: Either "steepest_descent" or "trust_regions"
        max_iterations: Maximum number of iterations
        verbose: 0 for quiet, 1 for summary, 2 for detailed output
        **optimizer_kwargs: Additional arguments passed to the optimizer
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
        Solve the relaxed approximate symmetry problem (rASP).

        Minimizes: -tr(AP A^T P^T - diag(c)P) over doubly stochastic matrices P.

        Args:
            A: Adjacency matrix of the graph (numpy array of shape (n, n))
            c: Weight vector (numpy array of shape (n,) or scalar)

        Returns:
            P_opt: Optimal doubly stochastic matrix (numpy array of shape (n, n))
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
        manifold = DoublyStochastic(n, retraction_method="simple")

        # Define cost function
        @pymanopt.function.pytorch(manifold)
        def cost(P):
            # Original objective
            AP = A @ P
            APAT = AP @ A.t()
            term1 = -torch.sum(APAT * P)
            #term1 = torch.mean(torch.pow(P @ A @ P.t() - A, 2))

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

        # Convert result back to numpy
        P_opt = result.point

        return {
            "P": P_opt, 
            "metrics": {
                "time": end_time - start_time,
                "iterations": result.iterations
            }
        }
