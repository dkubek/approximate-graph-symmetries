import time

import cyipopt
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, eye

from AGS.initialization import (
    check_random_state,
    init_barycenter,
    init_random_doubly_stochastic,
    init_random_permutation,
)


class InteriorPoint:
    """
    Minimizes: -tr(APA^T P^T) + tr(diag(c)P)
    Subject to: P1 = 1, P^T1 = 1, 0 <= P <= 1
    """

    def __init__(
        self,
        max_iter=1000,
        tol=1e-6,
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

        self._iteration_count = 0

    def _setup_sparsity_patterns(self):
        """Pre-compute sparsity patterns for Jacobian and Hessian."""

        self._setup_jacobian_sparsity()

        # Hessian sparsity pattern: A ⊗ A + A^T ⊗ A^T
        self._setup_hessian_sparsity()

    def _setup_jacobian_sparsity(self):
        n = self._n

        # Jacobian sparsity pattern
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

    def _setup_hessian_sparsity(self):
        """Compute sparsity pattern and values for the constant Hessian."""
        n = self._n
        
        hess_sparse = sp.kron(self.A, self.A) + sp.kron(self.A, self.A)
        
        # Convert to COO format for easy indexing
        hess_coo = hess_sparse.tocoo()
        
        # Extract lower triangular part
        lower_mask = hess_coo.row >= hess_coo.col
        
        self.hess_rows = hess_coo.row[lower_mask]
        self.hess_cols = hess_coo.col[lower_mask]
        self.hess_values = hess_coo.data[lower_mask]

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

    def hessian(self, x, lagrange, obj_factor):
        """Compute Hessian of the Lagrangian."""
        # Since constraints are linear, their Hessian contribution is zero
        # Hessian = obj_factor * (Hessian of objective)
        return obj_factor * self.hess_values

    def hessianstructure(self):
        """Return sparsity structure of Hessian."""
        return (self.hess_rows, self.hess_cols)

    def intermediate(
        self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
        d_norm, regularization_size, alpha_du, alpha_pr, ls_trials
    ):
        """IPOPT callback to track iterations."""
        self._iteration_count = iter_count

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

        # Pre-compute sparsity patterns and Hessian values
        self._setup_sparsity_patterns()

        # Reset iteration counter
        self._iteration_count = 0

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

        # Use exact Hessian instead of approximation
        nlp.add_option("hessian_approximation", "limited-memory")
        nlp.add_option("limited_memory_max_history", 10)
        nlp.add_option("hessian_constant", "yes")

        # Optimize for interior point method
        nlp.add_option("mu_strategy", "adaptive")
        nlp.add_option("mehrotra_algorithm", "yes")
        #nlp.add_option("linear_solver", "ma86")
        nlp.add_option("linear_solver", "mumps")

        # Solve
        start_time = time.time()
        x_opt, _ = nlp.solve(x0)
        end_time = time.time()

        # Reshape to matrix form
        P_opt = x_opt.reshape((n, n))

        return {
            "P": P_opt, 
            "metrics": {
                "time": end_time - start_time,
                "iterations": self._iteration_count
            }
        }