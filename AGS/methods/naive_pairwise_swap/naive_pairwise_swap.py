import numpy as np
from dataclasses import dataclass


@dataclass
class NaivePairwiseSwap:
    """
    Method: Naive Pairwise Swap (NPS)
    Type: Heuristic / Baseline
    Optimization Space: Set of all Transpositions (swaps of 2 nodes)
    Idea: Analytically finds the pair of nodes with the minimal
          neighborhood symmetric difference, using the corrected formula.
    """

    verbose: int = 0

    def solve(self, A: np.ndarray, c: float = None) -> dict:
        """
        Finds the best single transposition to minimize asymmetry.

        Args:
            A: Adjacency matrix (n, n), assumed symmetric and binary
            c: Penalty parameter (ignored, for interface compatibility)

        Returns:
            dict with keys:
                - 'P': Permutation matrix (n, n) as np.ndarray
                - 'metrics': dict with 'u', 'v', 'symmetric_difference', 'final_loss'
        """

        n = A.shape[0]
        D = A.sum(axis=1)
        M = A @ A

        # symmetric difference formula
        S = D[:, None] + D[None, :] - 2 * M - 2 * A
        np.fill_diagonal(S, np.inf)
        u, v = np.unravel_index(np.argmin(S), S.shape)

        # Construct permutation matrix for swapping u and v
        P = np.eye(n)
        P[[u, v]] = P[[v, u]]
        metrics = {
            "u": u,
            "v": v,
            "symmetric_difference": S[u, v],
            "final_loss": 0.25 * np.sum(np.abs(A - P @ A @ P.T)),
        }
        if self.verbose:
            print(
                f"NaivePairwiseSwap: swap nodes {u} and {v}, symmetric diff = {S[u, v]}"
            )
        return {"P": P, "metrics": metrics}
