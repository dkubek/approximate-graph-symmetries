import scipy as sp
import networkx as nx
import pandas as pd
from collections import namedtuple

import numpy as np
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

# from galahad import qpa


ProblemFormulation = namedtuple('ProblemFormulation', [
    'n',          # dimension
    'm',          # number of general constraints
    'H_ne',       # Hessian elements
    'H_row',      # row indices, NB lower triangle
    'H_col',      # column indices, NB lower triangle
    'H_val',      # values
    'g',          # linear term in the objective
    'f',          # constant term in the objective
    'rho_g',      # penalty parameter for general constraints
    'rho_b',      # penalty parameter for simple bound constraints
    'A_ne',       # Jacobian elements
    'A_row',      # row indices
    'A_col',      # column indices
    'A_val',      # values
    'c_l',        # constraint lower bound
    'c_u',        # constraint upper bound
    'x_l',        # variable lower bound
    'x_u'         # variable upper bound
])


def formulate(G: nx.Graph) -> ProblemFormulation:
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G)

    H = sp.sparse.kron(A, A) + sp.sparse.kron(A.transpose(), A)
    H = sp.sparse.tril(H)
    H = H.tocoo()

    H_ne = H.nnz
    H_row = H.row  # row indices, NB lower triangle
    H_col = H.col  # column indices, NB lower triangle
    H_val = H.data  # values

    A1 = sp.sparse.kron(np.ones(n), sp.sparse.eye(n))
    A2 = sp.sparse.kron(sp.sparse.eye(n), np.ones(n))
    A = sp.sparse.vstack([A1, A2])
    A = A.tocoo()

    A_ne = A.nnz
    A_row = A.row
    A_col = A.col
    A_val = A.data

    g = np.zeros(n * n)  # linear term in the objective
    f = 0  # constant term in the objective

    c_l = np.ones(2 * n)    # constraint lower bound
    c_u = np.ones(2 * n)    # constraint upper bound
    x_l = np.zeros(n * n)   # variable lower bound
    x_u = np.ones(n * n)    # variable upper bound

    inttype = np.int64
    prob = ProblemFormulation(
        n * n, 2 * n, H_ne, H_row.astype(inttype), H_col.astype(inttype), H_val.astype(float), g.astype(float), f, 0.1, 0.1, A_ne, A_row.astype(inttype), A_col.astype(inttype), A_val.astype(float), c_l.astype(float), c_u.astype(float),
        x_l.astype(float), x_u.astype(float)
    )

    return prob


def solve(P: ProblemFormulation, solver):
    # allocate internal data and set default options
    options = solver.initialize()

    # set some non-default options
    # options['print_level'] = 0
    # print("options:", options)

    # load data (and optionally non-default options)
    solver.load(
        P.n, P.m,
        'coordinate', P.H_ne, P.H_row, P.H_col, None,
        'coordinate', P.A_ne, P.A_row, P.A_col, None,
        options)

    #  provide starting values (not crucial)

    x = np.zeros(P.n)
    y = np.zeros(P.m)
    z = np.zeros(P.n)

    # find optimum of qp
    print("1st problem: solve qp")
    x, c, y, z, x_stat, c_stat \
        = solver.solve_qp(P.n, P.m, P.f, P.g, P.H_ne, P.H_val, P.A_ne, P.A_val,
                          P.c_l, P.c_u, P.x_l, P.x_u, x, y, z)
    print(" x:", x)
    print(" c:", c)
    print(" y:", y)
    print(" z:", z)
    print(" x_stat:", x_stat)
    print(" c_stat:", c_stat)

    # get information
    inform = solver.information()
    print(" f: %.4f" % inform['obj'])

    # deallocate internal data

    solver.terminate()


if __name__ == '__main__':
    G = nx.erdos_renyi_graph(10, 0.3)
    P = formulate(G)
    # solve(P)
