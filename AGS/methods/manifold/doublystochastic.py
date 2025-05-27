from typing import Optional, Tuple

import numpy as np
import scipy.sparse.linalg
from pymanopt.manifolds.manifold import Manifold


class DoublyStochastic(Manifold):
    r"""Manifold of doubly-stochastic matrices with positive entries.

    The manifold consists of n×n matrices X with positive entries such that each
    row and column sums to 1. The manifold has dimension (n-1)².

    The Riemannian metric is the Fisher information metric:
        <U, V>_X = sum_ij (U_ij * V_ij / X_ij)

    Args:
        n: Size of the square matrices.
        retraction_method: Method for retraction. Options are:
            - "simple": First-order retraction X + eta
            - "sinkhorn": Exponential-based with Sinkhorn projection (default)
        max_sinkhorn_iters: Maximum iterations for Sinkhorn-Knopp algorithm.
        pcg_threshold: Use PCG solver when n exceeds this threshold.

    Note:
        The implementation follows the embedded geometry described in [Douik2018]_.
        The retraction is only valid in a neighborhood of the current point where
        all entries remain positive.

    References:
        .. [Douik2018] A. Douik and B. Hassibi, "Manifold Optimization Over the Set
           of Doubly Stochastic Matrices: A Second-Order Geometry"
           ArXiv:1802.02628, 2018.
    """

    def __init__(
        self,
        n: int,
        *,
        retraction_method: str = "sinkhorn",
        max_sinkhorn_iters: Optional[int] = None,
        pcg_threshold: int = 100,
    ):
        self._n = n
        self._retraction_method = retraction_method
        self._max_sinkhorn_iters = max_sinkhorn_iters or (1000 + 2 * n)
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
        return np.sum(tangent_vector_a * tangent_vector_b / point)

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
        Y = point + tangent_vector
        # Ensure positivity first
        Y = np.maximum(Y, self._eps)
        # Project to doubly stochastic
        return self._doubly_stochastic(Y)

    def _retraction_sinkhorn(self, point, tangent_vector):
        """Exponential-based retraction with Sinkhorn projection."""
        # Clip to prevent overflow
        safe_point = np.maximum(point, self._eps)

        # Limit the magnitude to prevent exp overflow (exp(>700) overflows)
        ratio = tangent_vector / safe_point
        max_ratio = 50.0
        ratio = np.clip(ratio, -max_ratio, max_ratio)

        Y = point * np.exp(ratio)
        # Project to doubly stochastic
        return self._doubly_stochastic(Y)

    def random_point(self):
        """Generate random doubly stochastic matrix."""
        # Start with random positive matrix
        X = np.abs(np.random.randn(self._n, self._n)) + self._eps
        # Project to doubly stochastic
        return self._doubly_stochastic(X)

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

    def _linear_solve(self, point, b):
        n = self._n

        # Split b into components
        Z1 = b[:n]  # sum(Z, axis=1)
        ZT1 = b[n:]  # sum(Z, axis=0)

        # Compute (I - XX^T)
        I_minus_XXT = np.eye(n) - point @ point.T

        # Compute alpha using pseudo-inverse
        # alpha = (I - XX^T)^† (Z1 - X*Z^T*1)
        alpha = np.linalg.pinv(I_minus_XXT) @ (Z1 - point @ ZT1)

        # Compute beta
        beta = ZT1 - point.T @ alpha

        return alpha, beta

    def _doubly_stochastic(self, X, tol=1e-8):
        """Project matrix to doubly stochastic using Sinkhorn-Knopp algorithm."""
        X = np.maximum(X, self._eps)

        for iter_num in range(self._max_sinkhorn_iters):
            X_old = X.copy()

            # Row normalization
            row_sums = np.sum(X, axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, self._eps)
            X = X / row_sums

            # Column normalization
            col_sums = np.sum(X, axis=0, keepdims=True)
            col_sums = np.maximum(col_sums, self._eps)
            X = X / col_sums

            # More robust convergence check
            rel_change = np.max(np.abs(X - X_old)) / (1.0 + np.max(np.abs(X)))
            if rel_change < tol:
                break

        # Final safety check
        X = np.clip(X, 0, 1)
        return X

    def dist(self, point_a, point_b):
        """Geodesic distance between two points."""
        # This is an approximation; exact geodesic distance is complex
        return self.norm(point_a, self.log(point_a, point_b))

    def exp(self, point, tangent_vector):
        """Exponential map (approximation via retraction)."""
        # For this manifold, we use retraction as approximation
        return self.retraction(point, tangent_vector)

    def log(self, point_a, point_b):
        """Logarithmic map (approximation)."""
        # Project the difference to tangent space
        return self.projection(point_a, point_b - point_a)

    def pair_mean(self, point_a, point_b):
        """Compute the mean of two points."""
        # Simple approach: normalize the average
        mean = (point_a + point_b) / 2
        return self._doubly_stochastic(mean)
