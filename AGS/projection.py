from typing import Iterator, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def sinkhorn(
    X: np.ndarray, max_iter: Optional[int] = None, tol: float = 1e-9
) -> np.ndarray:
    """
    Apply Sinkhorn operator to make a matrix doubly stochastic.

    Parameters
    ----------
    X : np.ndarray
        Non-negative square matrix to normalize
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    P : np.ndarray
        Doubly stochastic matrix
    """

    n = X.shape[0]
    P = X.copy()

    max_iter = max_iter or 100 + 2 * n

    for _ in range(max_iter):
        # Row normalization
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        P = P / row_sums

        # Column normalization
        col_sums = P.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        P = P / col_sums

        # Check convergence
        if (
            np.abs(P.sum(axis=1) - 1).max() < tol
            and np.abs(P.sum(axis=0) - 1).max() < tol
        ):
            break

    return P


def project_to_permutation(D):
    """
    Project the double stochastic matrix D to the space of permutation matrices
    using Hungarian algorithm.
    """

    return linear_sum_assignment(-D)[1]


def birkhoff_decomposition(
    D: np.ndarray, tol: float = 1e-9
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Decompose a doubly stochastic matrix into a convex combination of permutation matrices.

    Uses the Birkhoff-von Neumann decomposition algorithm, which greedily extracts
    permutation matrices from the doubly stochastic matrix.

    Parameters
    ----------
    D : np.ndarray
        A doubly stochastic matrix (non-negative, rows and columns sum to 1)
    tol : float, optional
        Tolerance for considering matrix entries as zero

    Yields
    ------
    weight : float
        The weight of the permutation matrix in the convex combination
    perm_matrix : np.ndarray
        A permutation matrix (binary matrix with exactly one 1 per row and column)
    """
    n = D.shape[0]
    D_work = D.copy()

    while np.sum(D_work) > tol:
        row_ind, col_ind = linear_sum_assignment(-D_work)

        # Find the minimum positive value along this permutation
        perm_values = D_work[row_ind, col_ind]
        positive_mask = perm_values > tol

        if not np.any(positive_mask):
            break

        min_weight = np.min(perm_values[positive_mask])

        perm_matrix = np.zeros((n, n))
        perm_matrix[row_ind, col_ind] = 1

        yield min_weight, perm_matrix

        D_work -= min_weight * perm_matrix

        D_work = np.maximum(D_work, 0)


def perturb_doubly_stochastic(
    D: np.ndarray, variance: float = 0.01, random_state: Optional[int] = None
) -> np.ndarray:
    """
    Perturb a doubly stochastic matrix with Gaussian noise and project back.

    This implements the stochastic perturbation procedure.

    Parameters
    ----------
    D : np.ndarray
        A doubly stochastic matrix to perturb
    variance : float, optional
        Variance of the Gaussian noise. Default is 0.01, which is appropriate
        for matrices where entries are O(1/n). For n=100, average entry is 0.01,
        so this gives perturbations with std dev of 0.1.
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    D_perturbed : np.ndarray
        A perturbed doubly stochastic matrix
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Add Gaussian noise
    noise = np.random.normal(0, np.sqrt(variance), size=D.shape)
    D_noisy = D + noise

    D_noisy = np.clip(D_noisy, 0, 1)

    D_perturbed = sinkhorn(D_noisy)

    return D_perturbed


def randomized_recovery(
    D: np.ndarray,
    num_samples: int = 10,
    variance: float = 0.01,
    random_state: Optional[int] = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Generate multiple permutation decompositions by perturbing the input matrix.

    This implements the randomized recovery procedure mentioned in the paper excerpt,
    where multiple perturbations are used to get different decompositions.

    Parameters
    ----------
    D : np.ndarray
        A doubly stochastic matrix
    num_samples : int, optional
        Number of perturbations to generate
    variance : float, optional
        Variance for the perturbation
    random_state : int, optional
        Random seed for reproducibility

    Yields
    ------
    Results from birkhoff_decomposition for each perturbed matrix
    """
    rng = np.random.RandomState(random_state)

    for i in range(num_samples):
        D_perturbed = perturb_doubly_stochastic(
            D,
            variance=variance,
            random_state=rng.randint(2**31) if random_state is not None else None,
        )

        yield from birkhoff_decomposition(D_perturbed)
