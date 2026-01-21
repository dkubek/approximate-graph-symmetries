"""
Defines the manifold of doubly stochastic matrices.

This module provides the geometric definition of the manifold of n x n doubly
stochastic matrices, which is the set of matrices with positive entries where
each row and column sums to one. This manifold is the relative interior of the
Birkhoff polytope.

The implementation includes the definition of the manifold's tangent space,
the Fisher Information Metric, and the necessary geometric operations for
optimization, such as projection, retraction, and gradient computation.
These tools enable the use of Riemannian optimization algorithms on this space.

This implementation is adapted from the MATLAB `manopt` toolbox and the theoretical
framework described in the paper "Manifold optimization over the set of doubly
stochastic matrices: A second-order geometry" by Douik and Hassibi.

References:
    - Douik, A., & Hassibi, B. (2019). "Manifold optimization over the set
      of doubly stochastic matrices: A second-order geometry."
    - The `manopt` (MATLAB) toolbox: https://github.com/NicolasBoumal/manopt
"""

from typing import Optional

import logging
import numpy as np
from numba import jit
import scipy.sparse.linalg
from scipy.sparse.linalg import cg, lsqr, LinearOperator
from pymanopt.manifolds.manifold import Manifold


@jit(nopython=True)
def _doubly_stochastic(X, tol=1e-8, max_iters=100, eps=1e-8):
    """
    Sinkhorn-Knopp algorithm to project a matrix onto a Birkhoff polytope.
    """
    # Ensure a copy is made if the original array needs to be preserved
    X_norm = X.copy()
    X_norm = np.maximum(X_norm, eps)

    for iter_num in range(max_iters):
        # Row normalization
        row_sums = np.sum(X_norm, axis=1)
        row_sums = np.maximum(row_sums, eps)
        X_norm /= row_sums.reshape(-1, 1)

        # Column normalization
        col_sums = np.sum(X_norm, axis=0)
        col_sums = np.maximum(col_sums, eps)
        X_norm /= col_sums

        # Convergence check
        col_err = np.max(np.abs(1.0 - col_sums))
        row_err = np.max(np.abs(1.0 - np.sum(X_norm, axis=1)))

        if max(row_err, col_err) < tol:
            break

    return np.clip(X_norm, 0, 1)


class DoublyStochastic(Manifold):
    r"""
    Manifold of n x n doubly stochastic matrices with positive entries.

    This manifold consists of matrices where all entries are positive, and each
    row and column sums to 1. It represents the relative interior of the
    Birkhoff polytope and has a dimension of (n-1)².

    The Riemannian metric used is the Fisher information metric:
    `<U, V>_X = sum_ij (U_ij * V_ij / X_ij)`

    Note:
        This implementation is adapted from the MATLAB `manopt` toolbox, available
        at https://github.com/NicolasBoumal/manopt, and follows the embedded
        geometry described in [Douik2018]. The retraction is only valid in a
        neighborhood of the current point where all entries remain positive.

    Args:
        n (int): The size of the square matrices.
        retraction_method (str): The retraction method to use. Options are:
            "simple" for a first-order additive retraction or "sinkhorn" for an
            exponential-based retraction with Sinkhorn projection.
        max_sinkhorn_iters (int, optional): The maximum number of iterations for the
            Sinkhorn-Knopp algorithm used in retractions.
        pcg_threshold (int): A threshold for `n` above which a PCG solver is
            used for linear systems.

    References:
        .. [Douik2018] A. Douik and B. Hassibi, "Manifold Optimization Over the Set
           of Doubly Stochastic Matrices: A Second-Order Geometry"
           ArXiv:1802.02628, 2018.
    """

    def __init__(
        self,
        n: int,
        *,
        retraction_method: str = "simple",
        max_sinkhorn_iters: Optional[int] = None,
        pcg_threshold: int = 100,
    ):
        self._n = n
        self._retraction_method = retraction_method
        self._max_sinkhorn_iters = max_sinkhorn_iters or (100 + 2 * n)
        self._sinkhorn_tol = 1e-8
        self._pcg_threshold = pcg_threshold

        name = f"Doubly stochastic manifold DS({n})"
        dimension = (n - 1) ** 2
        super().__init__(name, dimension)

        # Precompute reusable arrays
        self._e = np.ones(n)
        # self._eps = np.finfo(np.float64).eps
        self._eps = 1e-8

    @property
    def typical_dist(self):
        # The manifold is not compact as a result of the choice of the metric,
        # thus any choice here is arbitrary. This is notably used to pick
        # default values of initial and maximal trust-region radius in the
        # trustregions solver.
        return self._n

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        """Fisher information metric."""
        safe_point = np.maximum(point, self._eps)
        return np.sum(tangent_vector_a * tangent_vector_b / safe_point)

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def projection(self, point, vector):
        """Project vector to tangent space at point."""
        # Compute b = [sum(Z, axis=1); sum(Z, axis=0)]
        b = np.concatenate([np.sum(vector, axis=1), np.sum(vector, axis=0)])

        # Solve linear system for alpha, beta
        alpha, beta = self._linear_solve(point, b)

        # Return projected vector
        eta = vector - point * (alpha[:, np.newaxis] + beta[np.newaxis, :])
        return eta

    to_tangent_space = projection

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        """Convert Euclidean to Riemannian gradient."""
        return self.projection(point, euclidean_gradient * point)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        """Convert Euclidean to Riemannian Hessian."""
        # Scale gradients and hessian
        gamma = euclidean_gradient * point
        gamma_dot = euclidean_hessian * point + euclidean_gradient * tangent_vector

        # Project gamma to get delta
        b = np.concatenate([np.sum(gamma, axis=1), np.sum(gamma, axis=0)])
        alpha, beta = self._linear_solve(point, b)
        delta = gamma - point * (alpha[:, np.newaxis] + beta[np.newaxis, :])

        # Compute directional derivative
        b_dot = np.concatenate([np.sum(gamma_dot, axis=1), np.sum(gamma_dot, axis=0)])

        # Adjust for tangent vector contribution
        adjustment = np.concatenate([tangent_vector @ beta, tangent_vector.T @ alpha])

        alpha_dot, beta_dot = self._linear_solve(point, b_dot - adjustment)

        # Compute nabla and project
        delta_dot = (
            gamma_dot
            - point * (alpha_dot[:, np.newaxis] + beta_dot[np.newaxis, :])
            - tangent_vector * (alpha[:, np.newaxis] + beta[np.newaxis, :])
        )

        nabla = delta_dot - 0.5 * (delta * tangent_vector) / point
        return self.projection(point, nabla)

    def retraction(self, point, tangent_vector):
        """Retract tangent vector to manifold."""

        if self._retraction_method == "simple":
            return self._retraction_simple(point, tangent_vector)
        elif self._retraction_method == "sinkhorn":
            return self._retraction_sinkhorn(point, tangent_vector)
        else:
            raise ValueError(f"Unknown retraction method: {self._retraction_method}")

    def _retraction_simple(self, point, tangent_vector):
        """Simple first-order retraction."""

        # Find maximum step size to preserve non-negativity
        negative_mask = tangent_vector < 0
        if np.any(negative_mask):
            ratios = -point[negative_mask] / tangent_vector[negative_mask]
            max_alpha = np.min(ratios)
        else:
            max_alpha = np.inf

        alpha = min(1, max_alpha)

        Y = point + alpha * tangent_vector
        return Y

    def _retraction_sinkhorn(self, point, tangent_vector, max_exp_arg=50.0):
        """Exponential-based retraction with Sinkhorn projection."""
        # Clip to prevent overflow
        safe_point = np.maximum(point, self._eps)

        # Limit the magnitude to prevent exp overflow (exp(>700) overflows)
        ratio = tangent_vector / safe_point
        ratio = np.clip(ratio, -max_exp_arg, max_exp_arg)

        Y = point * np.exp(ratio)
        # Project to doubly stochastic
        return _doubly_stochastic(
            Y, self._sinkhorn_tol, self._max_sinkhorn_iters, self._eps
        )

    def random_point(self):
        """Generate random doubly stochastic matrix."""
        # Start with random positive matrix
        X = np.abs(np.random.randn(self._n, self._n)) + self._eps
        # Project to doubly stochastic
        return _doubly_stochastic(
            X, self._sinkhorn_tol, self._max_sinkhorn_iters, self._eps
        )

    def random_tangent_vector(self, point):
        """Generate random tangent vector."""
        # Random matrix in ambient space
        Z = np.random.randn(self._n, self._n)
        # Project to tangent space and normalize
        eta = self.projection(point, Z)
        return eta / self.norm(point, eta)

    def zero_vector(self, point):
        return np.zeros((self._n, self._n))

    def transport(self, point_a, point_b, tangent_vector_a):
        """Transport tangent vector from point_a to point_b."""
        return self.projection(point_b, tangent_vector_a)

    def _linear_solve_pinv(self, point, b):
        n = self._n

        # Unpack the right-hand side vector b
        Z1 = b[:n]  # Corresponds to Z1 in the paper
        ZT1 = b[n:]  # Corresponds to Z^T 1 in the paper

        # The system for α is: (I - XX^T)α = Z1 - X(Z^T 1)
        # The matrix (I - XX^T) is singular, so we must use the pseudoinverse.
        I_minus_XXT = np.eye(n) - point @ point.T

        # Solve for alpha using pseudoinverse
        alpha = np.linalg.pinv(I_minus_XXT) @ (Z1 - point @ ZT1)

        # Solve for beta using alpha
        # β = Z^T 1 - X^T α
        beta = ZT1 - point.T @ alpha

        return alpha, beta

    def _linear_solve_cg(self, point, b):
        n = self._n

        # Define the matrix-vector product function for the block system
        # This function computes A @ x without ever forming A explicitly.
        def matvec(x):
            x_top = x[:n]
            x_bottom = x[n:]
            Ax_top = x_top + point @ x_bottom
            Ax_bottom = point.T @ x_top + x_bottom
            return np.concatenate([Ax_top, Ax_bottom])

        # Create a LinearOperator. This is the efficient way to use
        # SciPy's iterative solvers.
        A = LinearOperator((2 * n, 2 * n), matvec=matvec)

        # Use the conjugate gradient solver. It's fast and well-suited for this
        # symmetric positive-definite system.
        # 'zeta' will contain both alpha and beta stacked.
        zeta, info = cg(A, b)  # info=0 on success

        if info != 0:
            # Handle case where the cg solver did not converge
            logging.debug(
                f"Warning: Conjugate Gradient solver did not converge. Info: {info}"
            )
            logging.debug(f"Using pseudoinverse to solve the linear system.")
            return self._linear_solve_pinv(point, b)

        # Split the result back into alpha and beta
        alpha = zeta[:n]
        beta = zeta[n:]

        return alpha, beta

    def _linear_solve_lsqr(self, point, b):
        """
        Solves for α and β using the iterative LSQR algorithm.
        This is more efficient than pinv for very large, structured systems
        as it avoids forming the n x n system matrix.
        """
        n = self._n
        Z1 = b[:n]
        ZT1 = b[n:]

        # Define the matrix-vector product for A = (I - X @ X.T)
        # This function computes A @ v without forming A.
        def matvec(v):
            return v - point @ (point.T @ v)

        # Since A is symmetric, the transpose-vector product is the same.
        # A_op is a "virtual" representation of our matrix A.
        A_op = LinearOperator(
            shape=(n, n),
            matvec=matvec,
            rmatvec=matvec,  # rmatvec is for A.T @ v
            dtype=point.dtype,
        )

        # The right-hand side of the system A @ alpha = b_alpha
        b_alpha = Z1 - point @ ZT1

        # Use LSQR to solve the least-squares problem for alpha.
        # We can set a tolerance and iteration limit for performance tuning.
        # iter_lim is a reasonable default to prevent infinite loops.
        # atol and btol are standard stopping criteria.
        lsqr_result = lsqr(A_op, b_alpha, iter_lim=2 * n, atol=1e-8, btol=1e-8)
        alpha = lsqr_result[0]
        istop = lsqr_result[1]
        itn = lsqr_result[2]

        if istop > 2:
            logging.debug(
                f"LSQR solver for projection may not have converged. "
                f"Stop reason: {istop}, Iterations: {itn}"
            )

        # Once alpha is found, beta is computed the same way.
        beta = ZT1 - point.T @ alpha

        return alpha, beta

    def _linear_solve(self, point, b):
        return self._linear_solve_pinv(point, b)

    def dist(self, point_a, point_b):
        """Geodesic distance between two points."""
        # This is an approximation; exact geodesic distance is complex
        return self.norm(point_a, self.log(point_a, point_b))

    def exp(self, point, tangent_vector):
        """Exponential map (approximation via retraction)."""
        # For this manifold, we use retraction as approximation

        print("exp", tangent_vector)
        if np.isnan(np.sum(tangent_vector)):
            import pdb

            pdb.set_trace()

        return self.retraction(point, tangent_vector)

    def log(self, point_a, point_b):
        """Logarithmic map (approximation)."""
        # Project the difference to tangent space
        return self.projection(point_a, point_b - point_a)

    def pair_mean(self, point_a, point_b):
        """Compute the mean of two points."""
        # Simple approach: normalize the average
        mean = (point_a + point_b) / 2
        return _doubly_stochastic(
            mean, self._sinkhorn_tol, self._max_sinkhorn_iters, self._eps
        )
