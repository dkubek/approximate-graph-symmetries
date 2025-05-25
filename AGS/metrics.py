import numpy as np


def E(A, P):
    E = 1 / 4 * np.linalg.norm(A - P @ A @ P.T, "fro") ** 2
    return E


def S(A, P):
    n = P.shape[0]
    S = 4 * E(A, P) / (n * (n - 1))
    return S


def HammingDist(l1, l2):
    return np.sum(~(np.asarray(l1) == np.asarray(l2)) * 1)


def no_fixed_points(P):
    return np.trace(P)
