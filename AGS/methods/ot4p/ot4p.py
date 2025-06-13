from time import time

import numpy as np
import torch
from tqdm import tqdm

from AGS.annealing import get_annealing_tau

from .OT4P.ot4p import OT4P


class OT4P4AS:
    def __init__(
        self,
        max_iter=3000,
        initial_tau=0.7,
        final_tau=0.5,
        annealing_scheme="exponential",
        decay_steps=5000,
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
        self.device = self.device if self.device != "mps" else "cpu"
        #self.device = "cpu"
        if self.verbose:
            print(f"Using {self.device} device")

    def loss(self, A, P, c):
        return self._loss_norm(A, P, c)

    def _loss_trace(self, A, P, c):
        quadratic_term = -torch.trace(A @ P @ A.transpose(-2, -1) @ P.transpose(-2, -1))

        diag_elements = torch.diagonal(P, dim1=-2, dim2=-1)
        positive_diags = torch.pow(diag_elements, 2)
        penalty = torch.sqrt(torch.sum(c * positive_diags))

        return quadratic_term + penalty

    def _loss_norm(self, A, P, c):
        # return torch.sqrt(torch.sum(torch.pow(P @ X @ P.transpose(-2, -1) - Y, 2))) + penalty
        quadratic_term = torch.mean(torch.pow(P @ A @ P.transpose(-2, -1) - A, 2))

        diag_elements = torch.diagonal(P, dim1=-2, dim2=-1)
        positive_diags = torch.pow(diag_elements, 2)
        penalty = c * torch.sum(positive_diags, dim=-1)
        penalty = torch.sum(c * positive_diags)

        return quadratic_term + penalty

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

        model = OT4P(n).to(device=self.device, dtype=self.dtype)
        weightP = torch.nn.Parameter(
            torch.rand_like(A, device=self.device, dtype=self.dtype), requires_grad=True
        )
        optimizer = torch.optim.AdamW([weightP], lr=self.learning_rate)

        # Initialize tracking variables
        best_loss = float("inf")
        best_weight = weightP.clone().detach()
        best_iteration = 0

        pbar = tqdm(range(self.max_iter), disable=(self.verbose < 1))
        start_time = time()
        iterations_performed = 0

        for i in pbar:
            current_tau = get_annealing_tau(
                i,
                self.decay_steps,
                self.initial_tau,
                self.final_tau,
                scheme=self.annealing_scheme,
            )

            # Training step
            optimizer.zero_grad()
            perm_matrix = model(weightP, tau=current_tau)
            loss_val = self.loss(A, perm_matrix, c)
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_([weightP], max_norm=1.0)
            optimizer.step()

            # Compute validation loss
            with torch.no_grad():
                perm_matrix_val = model(weightP, tau=0)
                loss_val = self.loss(A, perm_matrix_val, c)

            # Check for improvement
            if loss_val < best_loss * (1 - self.min_rel_improvement):
                best_loss = loss_val
                best_weight = weightP.clone().detach()
                best_iteration = i + 1

            # Update model base
            model.update_base(weightP)

            # Update progress bar
            pbar.set_description(f"Loss: {loss_val.item():.6f}, Tau: {current_tau:.4f}")

            if loss_val < 1e-6:
                print(f"Converged to target threshold at iteration {i + 1}")
                break

        end_time = time()

        P_opt = model(best_weight, tau=0).cpu().numpy()
        
        return {
            "P": P_opt, 
            "metrics": {
                "time": end_time - start_time,
                "iterations": iterations_performed,
                "best_iteration": best_iteration
            }
        }
