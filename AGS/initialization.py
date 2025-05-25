import numpy as np
from typing import Optional, Union
from .projection import sinkhorn


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
    sinkhorn_iters: Optional[int] = None,
) -> np.ndarray:
    """Initialize with a random doubly stochastic matrix using Sinkhorn normalization."""
    X = rng.uniform(size=(n, n))
    return sinkhorn(X, max_iter=sinkhorn_iters)
