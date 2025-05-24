import numpy as np
from typing import Optional, Union


def check_random_state(
    seed: Optional[Union[int, np.random.RandomState, np.random.Generator]],
) -> Union[np.random.RandomState, np.random.Generator]:
    """Convert seed to a numpy RandomState or Generator instance."""
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(int(seed))
    if isinstance(seed, np.random.RandomState):
        return seed
    try:
        if isinstance(seed, np.random.Generator):
            return seed
    except AttributeError:
        pass
    raise ValueError(
        f"{seed!r} cannot be used to seed a numpy.random.RandomState instance"
    )


def sinkhorn(
    X: np.ndarray, max_iter: Union[int, None] = None, tol: float = 1e-9
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


def init_barycenter(n: int) -> np.ndarray:
    """Initialize with the barycenter (uniform doubly stochastic matrix)."""
    return np.ones((n, n)) / n


def init_random_permutation(
    n: int, rng: Union[np.random.RandomState, np.random.Generator]
) -> np.ndarray:
    """Initialize with a random permutation matrix."""
    perm = rng.permutation(n)
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1
    return P


def init_random_doubly_stochastic(
    n: int,
    rng: Union[np.random.RandomState, np.random.Generator],
    sinkhorn_iters: Union[int, None] = None,
) -> np.ndarray:
    """Initialize with a random doubly stochastic matrix using Sinkhorn normalization."""
    X = rng.uniform(size=(n, n))
    return sinkhorn(X, max_iter=sinkhorn_iters)
