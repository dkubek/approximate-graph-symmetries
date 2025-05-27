import time
from typing import Literal, Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from AGS.initialization import (check_random_state, init_barycenter,
                                init_random_doubly_stochastic,
                                init_random_permutation)


class QSA:
    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-8,
        rng: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        verbose=1,
    ):
        """
        max_iter : int
            Maximum number of Frank-Wolfe iterations
        tol : float
            Convergence tolerance for relative Frobenius norm
        rng : int, RandomState, Generator, or None
            Random state for reproducibility
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
        Solve the relaxed approximate symmetry problem using Frank-Wolfe algorithm.

        Parameters
        ----------
        A : np.ndarray
            Adjacency matrix of the graph (n x n)
        c : float or np.ndarray
            Diagonal penalty term. If scalar, uses c * ones(n)
        init : str
            Initialization method: "barycenter", "random_permutation", or "random_doubly_stochastic"
        P0 : np.ndarray, optional
            User-supplied initial doubly stochastic matrix

        Returns
        -------
        P : np.ndarray
            Final doubly stochastic matrix
        solve_time : float
            Solution time in seconds
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
        for iteration in pbar:
            # Store previous P for convergence check
            P_prev = P.copy()

            # Step 1: Compute gradient
            # ∇f(P) = -APA^T - A^T PA + diag(c)
            grad = -2 * A @ P @ A + np.diag(c)

            # Step 2: Solve linear subproblem
            # min_{Q ∈ D_n} tr(grad^T Q)
            # This is equivalent to a linear assignment problem
            _, col_ind = linear_sum_assignment(grad)
            Q = np.eye(n)[col_ind]

            # Step 3: Compute optimal step size
            # We need to minimize f(P + alpha * (Q - P)) for alpha in [0, 1]
            # This is a quadratic function in alpha
            R = Q - P

            # Compute coefficients for quadratic a * alpha^2 + b * alpha + constant
            # where g(alpha) = f(P + αR) = a * alpha^2 + b * alpha^2 + constant
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

        return {"P": P, "time": solve_time}
