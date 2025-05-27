import time

import cyipopt
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, eye, kron

from AGS.initialization import (check_random_state, init_barycenter,
                                init_random_doubly_stochastic,
                                init_random_permutation)


class InteriorPoint:
    """
    Minimizes: -tr(APA^T P^T) + tr(diag(c)P)
    Subject to: P1 = 1, P^T1 = 1, 0 <= P <= 1
    """

    def __init__(
        self,
        max_iter=1000,
        tol=1e-8,
        rng=None,
        verbose=True,
    ):
        """
        Initialize the QSA problem.

        Parameters:
        -----------
        A : array_like or sparse matrix
            Adjacency matrix (n x n)
        c : scalar or array_like
            Penalty parameter (scalar or vector of length n)
        """

        self.rng = check_random_state(rng)
        self.max_iter = max_iter
        self.tol = tol

        self.verbose = verbose

    def _setup_sparsity_patterns(self):
        """Pre-compute sparsity patterns for Jacobian."""

        n = self._n

        # Row indices for Jacobian non-zeros
        jac_rows = []
        jac_cols = []

        # Row sum constraints (first n constraints)
        for i in range(n):
            for j in range(n):
                jac_rows.append(i)
                jac_cols.append(i * n + j)

        # Column sum constraints (next n constraints)
        for j in range(n):
            for i in range(n):
                jac_rows.append(n + j)
                jac_cols.append(i * n + j)

        self.jac_rows = np.array(jac_rows)
        self.jac_cols = np.array(jac_cols)

    def objective(self, x):
        """Compute objective value: -tr(APA^T P^T) + tr(diag(c)P)"""
        P = x.reshape((self._n, self._n))

        # Compute -tr(APA^T P^T)
        # Using the identity: tr(AB) = sum(A * B^T)
        APAT = self.A @ P @ self.A
        term1 = -np.sum(APAT * P)

        # Compute tr(diag(c)P) = sum(c_i * P_ii)
        term2 = np.sum(self.c_diag.diagonal() * np.diag(P))

        return term1 + term2

    def gradient(self, x):
        """Compute gradient of objective function."""
        P = x.reshape((self._n, self._n))

        # Gradient of -tr(APA^T P^T) is -A^T P A^T - A P^T A
        grad_matrix = -2 * self.A @ P @ self.A

        # Add gradient of tr(diag(c)P)
        grad_matrix = grad_matrix + self.c_diag

        return grad_matrix.ravel()

    def constraints(self, x):
        """Compute constraint values: [P1 - 1; P^T 1 - 1]"""
        P = x.reshape((self._n, self._n))
        ones = np.ones(self._n)

        # Row sums - 1
        row_constraints = P @ ones - ones

        # Column sums - 1
        col_constraints = P.T @ ones - ones

        return np.concatenate([row_constraints, col_constraints])

    def jacobian(self, x):
        """Compute Jacobian of constraints (sparse format)."""
        # The Jacobian structure is fixed:
        # - First n rows: derivatives of row sum constraints
        # - Next n rows: derivatives of column sum constraints

        # All derivatives are 1.0 for the respective elements
        return np.ones(len(self.jac_rows))

    def jacobianstructure(self):
        """Return sparsity structure of Jacobian."""
        return (self.jac_rows, self.jac_cols)

    def solve(
        self,
        A,
        c,
        P0="random_doubly_stochastic",
    ):
        """
        Solve the QSA problem using IPOPT.

        Parameters:
        -----------
        P0 : array_like, optional
            Initial doubly stochastic matrix
        init : str
            Initialization method if P0 is None
        rng : RandomState, optional
            Random number generator
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print optimization progress

        Returns:
        --------
        P : ndarray
            Optimized doubly stochastic matrix
        info : dict
            Optimization information
        """

        # Store problem data
        n = A.shape[0]
        self._n = n

        # Convert A to sparse format for efficiency
        if sp.issparse(A):
            A = A.tocsr()
        else:
            A = csr_matrix(A)

        self.A = A

        # Handle c parameter
        if np.isscalar(c):
            self.c_diag = c * eye(n, format="csr")
        else:
            self.c_diag = diags(c, format="csr")

        # Pre-compute sparsity patterns
        self._setup_sparsity_patterns()

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

        x0 = P.ravel()

        # set up bounds: 0 <= p_ij <= 1
        lb = np.zeros(n * n)
        ub = np.ones(n * n)

        # Constraint bounds: all equality constraints = 0
        cl = np.zeros(2 * n)
        cu = np.zeros(2 * n)

        # Create IPOPT problem
        nlp = cyipopt.Problem(
            n=n * n, m=2 * n, problem_obj=self, lb=lb, ub=ub, cl=cl, cu=cu
        )

        # Set IPOPT options
        nlp.add_option("max_iter", self.max_iter)
        nlp.add_option("tol", self.tol)

        if not self.verbose:
            nlp.add_option("print_level", 0)
            nlp.add_option("sb", "yes")

        # Use L-BFGS for Hessian approximation (simpler and often sufficient)
        nlp.add_option("hessian_approximation", "limited-memory")

        # Optimize for interior point method
        nlp.add_option("mu_strategy", "adaptive")
        nlp.add_option("mehrotra_algorithm", "yes")

        # Solve
        start_time = time.time()
        x_opt, _ = nlp.solve(x0)
        end_time = time.time()

        # Reshape to matrix form
        P_opt = x_opt.reshape((n, n))

        return {"P": P_opt, "time": end_time - start_time}
