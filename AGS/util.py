import numpy as np


def to_permutation_matrix(pi):
    """
    Construct permutation matrix from permutation vector.
    """

    return np.eye(len(pi))[pi]


def to_permutation_vector(P):
    """
    Construct a permutation vector from permutation matrix.
    """

    return np.array([np.where(P[i])[0] for i in range(P.shape[0])])[:, 0]
