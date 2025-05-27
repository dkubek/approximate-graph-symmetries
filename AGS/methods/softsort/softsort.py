from time import time

import numpy as np
import torch
from tqdm import tqdm

from AGS.annealing import get_annealing_tau


def soft_sort(s, tau):
    """
    PyTorch implementation of SoftSort function for 1D tensors.

    Args:
        s: Input tensor of shape (1,)
        tau: Temperature parameter for SoftSort

    Returns:
        P_hat: Soft permutation matrix of shape (n, n)
    """

    # Check that input is 1D
    if s.dim() != 1:
        raise ValueError(
            f"Expected 1D tensor of shape [sequence_length,], got tensor with {s.dim()} dimensions"
        )

    # Sort the tensor in descending order
    s_sorted, _ = torch.sort(s, dim=0, descending=True)

    # Reshape tensors for computing pairwise distances
    # Convert s to shape [sequence_length, 1] and s_sorted to shape [1, sequence_length]
    s_col = s.unsqueeze(1)  # Shape: [sequence_length, 1]
    s_sorted_row = s_sorted.unsqueeze(0)  # Shape: [1, sequence_length]

    # Compute pairwise distances
    # Broadcasting will create a tensor of shape [sequence_length, sequence_length]
    pairwise_distances = -torch.abs(s_col - s_sorted_row)

    # Apply softmax along the last dimension (dim=1 for a 2D tensor)
    P_hat = torch.nn.functional.softmax(pairwise_distances / tau, dim=1)

    return P_hat


class SoftSort:
    def __init__(
        self,
        max_iter=2000,
        initial_tau=1,
        final_tau=1e-6,
        annealing_scheme="cosine",
        decay_steps=None,
        learning_rate=1e-1,
        min_rel_improvement=1e-4,
        verbose=1,
    ):
        """
        max_iterations: Maximum number of iterations
        initial_tau: Starting temperature for annealing
        final_tau: Final temperature for annealing
        annealing_scheme: Method for annealing ("cosine", "linear", or "exponential")
        decay_rate: Rate of decay for exponential annealing
        decay_steps: Number of steps for complete annealing
        learning_rate: Learning rate for optimizer
        patience: Number of iterations to wait for improvement
        min_rel_improvement: Minimum relative improvement to reset patience
        """

        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.annealing_scheme = annealing_scheme
        self.decay_steps = decay_steps or max_iter

        self.min_rel_improvement = min_rel_improvement

        self.verbose = verbose

        self.dtype = torch.float
        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        if self.verbose:
            print(f"Using {self.device} device")

    def loss(self, A, P, c):
        diag_elements = torch.diagonal(P, dim1=-2, dim2=-1)
        penalty = torch.sum(c * diag_elements, dim=-1)
        # return torch.sqrt(torch.sum(torch.pow(P @ X @ P.transpose(-2, -1) - Y, 2))) + penalty
        return torch.mean(torch.pow(P @ A @ P.transpose(-2, -1) - A, 2)) + penalty

    def solve(
        self,
        A,
        c=0.2,
    ):
        """
        Optimize permutation matrix using soft sort with constraints.

        Args:
            A: Input matrix for optimization
            c: Penalization parameter for the loss function

        Returns:
            Tuple of (best_parameters, best_loss)
        """

        n = A.shape[0]

        # Handle scalar c
        if np.isscalar(c):
            c = np.full(n, c)
        else:
            c = np.asarray(c)

        # Validate inputs
        assert A.shape == (n, n), "A must be square"
        assert c.shape == (n,), f"c must have shape ({n},), got {c.shape}"

        # Convert to PyTorch tensors
        A = torch.from_numpy(A.astype(float)).to(device=self.device, dtype=self.dtype)
        c = torch.from_numpy(c.astype(float)).to(device=self.device, dtype=self.dtype)

        # Initialize s on the unit sphere
        s = torch.nn.Parameter(
            torch.nn.functional.normalize(
                torch.rand(n, device=self.device, dtype=self.dtype), dim=0
            ),
            requires_grad=True,
        )
        optimizer = torch.optim.AdamW([s], lr=self.learning_rate)

        # Initialize tracking variables
        best_loss = float("inf")
        best_s = s

        pbar = tqdm(range(self.max_iter), disable=(self.verbose < 1))
        start_time = time()
        for i in pbar:
            # Compute current tau using the annealing function
            current_tau = get_annealing_tau(
                i,
                self.decay_steps,
                self.initial_tau,
                self.final_tau,
                scheme=self.annealing_scheme,
            )

            # Training step
            optimizer.zero_grad()
            perm_matrix = soft_sort(s, tau=current_tau)
            loss_train = self.loss(A, perm_matrix, c=c)
            loss_train.backward()
            optimizer.step()

            # Compute validation loss
            with torch.no_grad():
                perm_matrix_val = soft_sort(s, tau=1e-8)
                loss_val = self.loss(A, perm_matrix_val, c=0)

            # Check for improvement
            if loss_val < best_loss * (1 - self.min_rel_improvement):
                best_loss = loss_val
                best_s = s.clone().detach()

            # Update progress bar
            pbar.set_description(f"Loss: {loss_val.item():.6f}, Tau: {current_tau:.4f}")

            if loss_val < 1e-5:
                print(f"Converged to target threshold at iteration {i + 1}")
                break

        end_time = time()

        P_opt = soft_sort(best_s, tau=1e-8).cpu().numpy()
        return {"P": P_opt, "time": end_time - start_time}
