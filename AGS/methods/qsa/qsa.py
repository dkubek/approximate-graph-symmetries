"""
Implements the Quadratic Symmetry Approximator (QSA) method.

This module provides a solver for the Relaxed Approximate Symmetry Problem
(rASP) using the Frank-Wolfe algorithm, also known as the conditional
gradient algorithm. QSA represents the state-of-the-art first-order method
for this problem.

The algorithm operates on the Birkhoff polytope (the set of doubly
stochastic matrices). At each iteration, it linearizes the quadratic objective
function and solves a Linear Assignment Problem (LAP) to find the descent
direction. This makes the method particularly efficient, as the LAP can be
solved in polynomial time. The final doubly stochastic matrix is the result
of a convex combination of permutation matrices found during the iterations.

The objective function includes a penalty term on the diagonal of the matrix
to discourage convergence to the trivial identity solution.

References:
    - Pidnebesna, A., Hartman, D., et al. (2025). "Computing Approximate
      Global Symmetry of Complex Networks..."
    - Vogelstein, J. T., Conroy, J. M., et al. (2015). "Fast approximate
      quadratic programming for graph matching."
    - Frank, M., & Wolfe, P. (1956). "An algorithm for quadratic programming."
"""

import time
from typing import Literal, Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from AGS.initialization import (
    check_random_state,
    init_barycenter,
    init_random_doubly_stochastic,
    init_random_permutation,
)


class QSA:
    """
    Solves the rASP using the Frank-Wolfe (conditional gradient) algorithm.

    This class implements the Quadratic Symmetry Approximator (QSA). It finds an
    approximate graph symmetry by minimizing a penalized quadratic objective
    function over the convex set of doubly stochastic matrices.
    """

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-8,
        rng: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        verbose=1,
    ):
        """
        Initializes the QSA solver.

        Args:
            max_iter (int): The maximum number of Frank-Wolfe iterations.
            tol (float): The tolerance for the change in the Frobenius norm of
                the solution matrix, used for checking convergence.
            rng (Union[int, np.random.RandomState, np.random.Generator], optional):
                A seed or random number generator for reproducible results,
                primarily for initialization. Defaults to None.
            verbose (int): Verbosity level. 0 for silent, 1 for progress bar.
        """

        self.max_iter = max_iter
        self.tol = tol
        self.rng = check_random_state(rng)

        self.verbose = verbose

    def solve(
        self,
        A: np.ndarray,
        c: Union[float, np.ndarray],
        P0: Union[
            Literal["barycenter", "random_permutation", "random_doubly_stochastic"],
            np.ndarray,
        ] = "random_doubly_stochastic",
    ):
        """
        Executes the Frank-Wolfe algorithm to solve the rASP.

        Args:
            A (np.ndarray): The n x n adjacency matrix of the graph.
            c (Union[float, np.ndarray]): The penalty parameter for the diagonal
                (fixed points). Can be a scalar or a vector of length n.
            P0 (Union[str, np.ndarray]): The initial doubly stochastic matrix P.
                Can be a specific numpy array or a string specifying an
                initialization method: "barycenter", "random_permutation", or
                "random_doubly_stochastic".

        Returns:
            dict: A dictionary containing:
                - "P" (np.ndarray): The final doubly stochastic matrix solution.
                - "metrics" (dict): A dictionary with performance metrics,
                  including 'time' and 'iterations'.
        """

        # Input validation
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError("A must be a square matrix")

        # Handle c parameter
        if np.isscalar(c):
            c = np.full(n, c)
        else:
            c = np.asarray(c)
            if c.shape != (n,):
                raise ValueError(f"c must be a scalar or vector of length {n}")

        # Initialize P
        if P0 is not None and isinstance(P0, np.ndarray):
            P = P0.copy()
            # Verify it's doubly stochastic
            if (
                P.shape != (n, n)
                or np.abs(P.sum(axis=1) - 1).max() > 1e-6
                or np.abs(P.sum(axis=0) - 1).max() > 1e-6
                or (P < 0).any()
            ):
                raise ValueError("P0 must be a doubly stochastic matrix")
        else:
            if P0 == "barycenter":
                P = init_barycenter(n)
            elif P0 == "random_permutation":
                P = init_random_permutation(n, self.rng)
            elif P0 == "random_doubly_stochastic":
                P = init_random_doubly_stochastic(n, self.rng)
            else:
                raise ValueError(f"Unknown initialization method: {P0}")

        # Frank-Wolfe iterations
        pbar = tqdm(range(self.max_iter), disable=(self.verbose < 1))
        start_time = time.time()
        iterations_performed = 0

        for iteration in pbar:
            iterations_performed = iteration + 1

            # Store previous P for convergence check
            P_prev = P.copy()

            # Step 1: Compute gradient
            # grad f(P) = -APA^T - A^T PA + diag(c)
            grad = -2 * A @ P @ A + np.diag(c)

            # Step 2: Solve linear subproblem
            # min_{Q in D_n} tr(grad^T Q)
            # This is equivalent to a linear assignment problem
            _, col_ind = linear_sum_assignment(grad)
            Q = np.eye(n)[col_ind]

            # Step 3: Compute optimal step size
            # We need to minimize f(P + alpha * (Q - P)) for alpha in [0, 1]
            # This is a quadratic function in alpha
            R = Q - P

            # Compute coefficients for quadratic a * alpha^2 + b * alpha + constant
            # where g(alpha) = f(P + alpha*R) = a * alpha^2 + b * alpha^2 + constant
            a = -np.trace(A @ R @ A @ R.T)
            b = (
                -np.trace(A @ P @ A @ R.T)
                - np.trace(A @ R @ A @ P.T)
                + np.trace(np.diag(c) @ R)
            )

            if abs(a) < 1e-8:
                # Case 1: a is close to zero (g is linear or constant: g(alpha) = b*alpha + k)
                # If b > 0, g(alpha) is increasing, min at alpha = 0.
                # If b < 0, g(alpha) is decreasing, min at alpha = 1.
                # If b = 0, g(alpha) is constant (g(alpha) = constant).
                alpha = 0.0 if b > 0 else 1.0

            elif a > 0:
                # Case 2: a > 0 (parabola opens upwards)
                # The unconstrained minimum is at the vertex alpha_v = -b / (2*a).
                # If alpha_v is in [0,1], that's our alpha.
                # If alpha_v < 0, the minimum on [0,1] is at alpha = 0.
                # If alpha_v > 1, the minimum on [0,1] is at alpha = 1.
                alpha_vertex = -b / (2.0 * a)
                alpha = np.clip(alpha_vertex, 0.0, 1.0)

            else:  # a < 0
                # Case 3: a < 0 (parabola opens downwards)
                # The minimum on a closed interval [0,1] must be at one of the endpoints.
                # We need to compare g(0) and g(1).
                # g(0) = k
                # g(1) = a*1^2 + b*1 + k = a + b + k
                # We want the alpha corresponding to min(g(0), g(1)).
                # If g(0) <= g(1)  =>  k <= a + b + k  =>  0 <= a + b, then alpha = 0.
                # Else (g(0) > g(1) => 0 > a + b), then alpha = 1.
                alpha = 0.0 if (a + b) > 0 else 1.0

            # Step 4: Update P
            P = P + alpha * R

            # Check convergence
            change = np.linalg.norm(P - P_prev, "fro") / np.sqrt(n)
            if change < self.tol:
                break

            # Update progress bar
            pbar.set_description(f"Change: {change:.6f}")

        solve_time = time.time() - start_time

        return {
            "P": P,
            "metrics": {"time": solve_time, "iterations": iterations_performed},
        }
