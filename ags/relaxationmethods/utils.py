import numpy as np
import networkx as nx
import itertools 

def is_symmetric(G: nx.Graph):
    n = G.number_of_nodes()

    pis = itertools.permutations(range(n))
    # Skip the identity
    next(pis)

    A = nx.to_numpy_array(G)
    print(G.adjacency)
    for pi in pis:
        P = permutation_to_matrix(pi)
        if np.array_equal(A @ P, P @ A):
            return True

    return False


def generate_asymmetric_graph(n, p=0.5):
    print(f"Generating random asymmetric graph n={n} p={p}")
    count = 0
    while True:
        print(f"{count}")
        G = nx.erdos_renyi_graph(n, p)

        print(G)
        if is_symmetric(G):
            count += 1
            continue

        break

    return G


def permutation_to_matrix(pi):
    n = len(pi)
    Im = np.eye(n)
    return Im[:, pi]


def random_permutation_matrix(n):
    # TODO: Use seed
    pi = np.random.permutation(n)
    return permutation_to_matrix(pi)


def doubly_stochastic(P: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (
            np.abs(P_eps.sum(axis=0) - 1) < tol
        ).all():
            # All column/row sums ~= 1 within threshold
            break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c

    return P_eps


def _compE(A, P):
    return 1 / 4 * np.linalg.norm(A - P @ A @ P.T, "fro") ** 2


def _compS(A, P, E=None):
    if E is None:
        E = _compE(A, P)

    n = int(np.sqrt(P.size))
    return 4 * E / (n * (n - 1))


def compES(A, perm):
    """
    Compute E(A) (1.1) and S(A) (1.2)
    """

    P = permutation_to_matrix(perm)
    N = len(perm)
    E = _compE(A, P)
    S = _compS(A, P, E=E)
    return E, S
