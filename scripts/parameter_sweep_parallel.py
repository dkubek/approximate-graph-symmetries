#!/usr/bin/env python3
"""
Parameter sweep script with parallel execution support.
"""

import sys
import itertools
import subprocess
import yaml
import time
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


def create_config_file(params: Dict[str, Any], output_path: Path) -> None:
    """Create a YAML configuration file with specified parameters."""
    with open(output_path, "w") as f:
        yaml.dump(params, f, default_flow_style=False)


def create_log_directory(method: str) -> Path:
    """Create a timestamped directory for log files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/sweep_{method}_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_param_string(param_names: List[str], combination: Tuple) -> str:
    """Create a descriptive string from parameter names and values."""
    param_str = "_".join(
        f"{name}={value}" for name, value in zip(param_names, combination)
    )
    # Replace problematic characters for filenames
    param_str = param_str.replace("/", "-").replace(" ", "_")
    return param_str


def run_parallel(
    jobs: List[Tuple[str, Path, Path]], max_workers: int
) -> Tuple[int, int]:
    """
    Run jobs in parallel.

    Args:
        jobs: List of (config_file, log_file, command) tuples
        max_workers: Maximum number of parallel workers

    Returns:
        Tuple of (successful_count, failed_count)
    """
    active_processes = {}
    job_queue = list(jobs)
    total_jobs = len(jobs)
    completed = 0
    successful = 0
    failed = 0

    print(f"\nRunning {total_jobs} combinations with {max_workers} parallel workers...")

    while job_queue or active_processes:
        # Start new processes if we have capacity
        while len(active_processes) < max_workers and job_queue:
            config_file, log_file, cmd = job_queue.pop(0)

            # Run command with output redirected to log file
            with open(log_file, "w") as f:
                proc = subprocess.Popen(
                    cmd, stdout=f, stderr=subprocess.STDOUT, text=True, shell=True
                )
            active_processes[proc] = (config_file, log_file)

        # Check for completed processes
        for proc in list(active_processes.keys()):
            if proc.poll() is not None:
                config_file, log_file = active_processes.pop(proc)
                completed += 1

                if proc.returncode == 0:
                    successful += 1
                    # Clean up config file on success
                    config_file.unlink()
                else:
                    failed += 1
                    print(f"  Failed: {config_file.name} (see {log_file})")

                # Progress update
                print(
                    f"  Progress: {completed}/{total_jobs} completed "
                    f"({successful} successful, {failed} failed)",
                    end="\r",
                )

        # Brief sleep to avoid busy waiting
        if active_processes:
            time.sleep(0.1)

    print()  # New line after progress updates
    return successful, failed


def run_parameter_sweep(
    method: str,
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    data_path: Path = Path("data/pidnebesna"),
    parallel: int = 1,
) -> None:
    """
    Run a parameter sweep for a specific method.

    Args:
        method: Method name (e.g., "InteriorPoint")
        base_config: Base configuration dict
        param_grid: Dictionary of parameter names to lists of values to try
        data_path: Path to data directory
        parallel: Number of parallel processes
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"\nParameter sweep for {method}:")
    print(f"Parameters: {param_names}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Parallel workers: {parallel}")

    # Create log directory
    log_dir = create_log_directory(method)
    print(f"Log directory: {log_dir}")

    # Prepare all jobs
    jobs = []
    start_time = time.time()

    for i, combination in enumerate(combinations):
        # Create config for this combination
        config = base_config.copy()

        # Update method-specific parameters
        if method not in config:
            config[method] = {}

        for param_name, param_value in zip(param_names, combination):
            if param_name in ["c", "num_runs", "num_workers"]:
                # Global parameters
                config[param_name] = param_value
            else:
                # Method-specific parameters
                config[method][param_name] = param_value

        # Create config file with descriptive name
        param_str = get_param_string(param_names, combination)
        config_file = Path(f"config_{method}_{param_str}.yaml")
        create_config_file(config, config_file)

        # Build command as a single string (simpler)
        cmd = f"python run.py --methods {method} --config {config_file} --data-path {data_path} --results-path test/"

        # Create log file path
        log_file = log_dir / f"run_{i:03d}_{param_str}.log"

        jobs.append((config_file, log_file, cmd))

    # Run jobs in parallel
    successful, failed = run_parallel(jobs, parallel)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Sweep completed in {elapsed:.1f} seconds")
    print(f"Successful runs: {successful}/{len(jobs)}")
    if failed > 0:
        print(f"Failed runs: {failed} (check log files in {log_dir})")
    print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Base configuration (shared across all runs)
    base_config = {"num_runs": 5, "num_workers": 10}

    # Example parameter sweep
    c_space = [0.7, 0.85, 1] #list(map(float, (np.linspace(0, 1, 21)[1:])))
    param_grids = {
        #"QSA": {
        #    "c": c_space,
        #    "max_iter": [200],
        #    # "initial_tau": [0.3, 0.5, 0.7],
        #    # "final_tau": [None],
        #},
        #"Manifold": {
        #    "c": c_space,
        #    "max_iter": [200],
        #},
        "InteriorPoint": {
            "c": c_space,
            "max_iter": [500],
        },
    }
    for method, param_grid in param_grids.items():
        run_parameter_sweep(
            method=method,
            base_config=base_config,
            param_grid=param_grid,
            parallel=1,
        )
