import math


def _calculate_progress(current_step: int, total_decay_steps: int) -> float:
    """
    Calculates the progress of annealing, clamped between 0.0 and 1.0.

    Args:
        current_step: The current step in the annealing process.
        total_decay_steps: The total number of steps over which annealing occurs.

    Returns:
        The progress as a float between 0.0 and 1.0.
    """
    if total_decay_steps <= 0:
        # If no decay steps, consider progress 0 at or before step 0, and 1 otherwise.
        # This means tau will be initial_tau if current_step <= 0, and final_tau if current_step > 0.
        return 0.0 if current_step <= 0 else 1.0

    # Calculate progress
    progress = float(current_step) / total_decay_steps

    # Clamp progress to [0, 1]
    return min(1.0, max(0.0, progress))


def _linear_anneal(initial_val: float, final_val: float, progress: float) -> float:
    """Linear interpolation between initial_val and final_val."""
    return (1.0 - progress) * initial_val + progress * final_val


def _exponential_anneal(initial_val: float, final_val: float, progress: float) -> float:
    """
    Exponential (geometric) interpolation.
    This function assumes initial_val and final_val are non-negative,
    as enforced by the caller `get_annealing_tau`.
    """
    # Handle boundary conditions explicitly for clarity and to ensure targets are met.
    if progress == 0.0:
        return initial_val
    if progress == 1.0:
        return final_val

    # At this point, progress is in (0, 1).
    # initial_val and final_val are guaranteed to be non-negative by the caller.

    if initial_val == 0.0:
        # If initial_val is 0, multiplicatively it stays 0.
        # (e.g., for 0 < progress < 1, if final_val > 0)
        return 0.0

    if final_val == 0.0:
        # initial_val > 0 here. Decays to 0.
        # initial_val * (0 / initial_val) ** progress = initial_val * 0 ** progress
        # math.pow(0.0, positive_progress) is 0.0.
        return initial_val * math.pow(
            0.0, progress
        )  # This will be 0.0 for progress in (0,1)

    # Both initial_val and final_val are > 0. progress is in (0,1).
    return initial_val * math.pow(final_val / initial_val, progress)


def _cosine_anneal(initial_val: float, final_val: float, progress: float) -> float:
    """Cosine annealing schedule."""
    return final_val + 0.5 * (initial_val - final_val) * (
        1.0 + math.cos(math.pi * progress)
    )


_SCHEME_MAP = {
    "linear": _linear_anneal,
    "exponential": _exponential_anneal,
    "cosine": _cosine_anneal,
}


def get_annealing_tau(
    i: int,
    decay_steps: int,
    initial_tau: float = 0.7,
    final_tau: float = 0.1,
    scheme: str = "linear",
) -> float:
    """
    Computes the annealing temperature (tau) for a given step using various schedules.

    This function calculates an intermediate value (often a temperature in simulated
    annealing or a learning rate in training) that changes over a number of steps
    from an initial value to a final value.

    Args:
        i: Current iteration or step number. Typically starts from 0.
        decay_steps: The total number of steps over which the value should transition
                     from initial_tau to final_tau.
        initial_tau: The starting value of tau (at step 0 or before).
        final_tau: The target value of tau (at step decay_steps or after).
        scheme: The annealing schedule to use. Supported options are:
                - "linear": Linear interpolation from initial_tau to final_tau.
                - "exponential": Exponential (geometric) decay/growth from
                                 initial_tau to final_tau. Requires non-negative
                                 initial_tau and final_tau.
                - "cosine": Cosine annealing, following half a cosine wave, from
                            initial_tau to final_tau.

    Returns:
        The computed annealing temperature (tau) for the current step `i`.

    Raises:
        ValueError: If an unknown `scheme` is provided.
        ValueError: If `scheme="exponential"` and `initial_tau` or `final_tau`
                    are negative.
    """
    if scheme not in _SCHEME_MAP:
        raise ValueError(
            f"Unknown scheme: '{scheme}'. Available schemes: {list(_SCHEME_MAP.keys())}"
        )

    if scheme == "exponential":
        if initial_tau < 0 or final_tau < 0:
            raise ValueError(
                "Exponential_scheme requires non-negative initial_tau and final_tau."
            )

    progress = _calculate_progress(i, decay_steps)

    anneal_func = _SCHEME_MAP[scheme]
    return anneal_func(initial_tau, final_tau, progress)
