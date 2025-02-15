import numpy as np
import scipy as sp

from approximate_graph_symmetries.relaxationmethods.utils import doubly_stochastic

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize


class ObjectiveFunction:
    def __init__(self, instance):
        self._instance = instance

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Objective function call not implemented")

    def jacobian(self):
        raise NotImplementedError("Jacobian not defined")

    def hessian(self):
        raise NotImplementedError("Hessian not defined")


class NonConvexRelaxation(ObjectiveFunction):
    def __call__(self, x):
        n = self._instance.n
        A = self._instance.adjacency
        P = x.reshape(n, n)
        return -np.trace(A @ P @ A.T @ P.T)


class NonConvexRelaxationPenalized(ObjectiveFunction):
    def __init__(self, instance, c):
        self._instance = instance

        self.c_mat = np.diag(np.ones(instance.n) * c)

    def __call__(self, x):
        n = self._instance.n
        A = self._instance.adjacency
        P = x.reshape(n, n)
        return -np.trace(A @ P @ A.T @ P.T - self.c_mat @ P)


class DynamicFixedPointsGMP(ObjectiveFunction):
    def __init__(self, instance):
        self._instance = instance

    def __call__(self, x):
        n = self._instance.n
        A = self._instance.adjacency
        P = x.reshape(n, n)
        E = np.linalg.norm(A - P @ A @ P.T, "fro")

        # Relaxed notion of a fixed point
        F = np.trace(P)

        return E / (n * (n - 1) - 2 * F * (F - 1))


class TrustRegionSolver:

    def __init__(
        self,
        init_method="uniform",
    ):
        assert init_method in ("uniform", "random")

        self._init_method = init_method

        self._instance = None
        self._formulated = False

        self._dimension = None
        self._constraint_matrix = None
        self._rhs = None
        self._lower_bounds = None
        self._upper_bounds = None

    def formulate(self, instance):
        self._instance = instance
        self._formulated = False

        self._dimension = instance.n * instance.n

        M, b = self._formulate_constraints()

        self._constraint_matrix = M
        self._rhs = b
        self._lower_bounds = np.zeros(self._dimension)
        self._upper_bounds = np.ones(self._dimension)

        self._formulated = True

    def _formulate_constraints(self):
        n = self._instance.n

        # Constraints
        ## Row sums
        M1 = sp.sparse.kron(np.ones(n), sp.sparse.eye(n))
        b1 = np.ones(n)

        ## Columns sums
        M2 = sp.sparse.kron(sp.sparse.eye(n), np.ones(n))
        b2 = np.ones(n)

        # Constraint Matrix
        M = sp.sparse.vstack([M1, M2])

        # RHS
        b = np.hstack([b1, b2])

        return M, b

    def _initial_point(self):
        assert self._formulated

        if self._init_method == "uniform":
            return np.ones(self._dimension) / self._instance.n
        elif self._init_method == "random":
            n = self._instance.n
            S = np.random.random((n, n))
            return doubly_stochastic(S)
        else:
            raise NotImplementedError()

    def solve(self, obj, callback=None):
        assert self._formulated

        linear_constraints = LinearConstraint(
            self._constraint_matrix, self._rhs, self._rhs
        )
        variable_bounds = Bounds(self._lower_bounds, self._upper_bounds)

        P_0 = self._initial_point()
        x_0 = P_0.flatten()

        res = minimize(
            obj,
            x_0,
            method="trust-constr",
            constraints=[linear_constraints],
            bounds=variable_bounds,
            options={"verbose": 1},
            callback=callback,
        )
        self._result = res

        return res
