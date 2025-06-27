#!/usr/bin/env python3
"""
Example script showing how to perform parameter sweeps with the new directory structure.
"""

import sys
import itertools
import subprocess
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def create_config_file(params: Dict[str, Any], output_path: Path) -> None:
    """Create a YAML configuration file with specified parameters."""
    with open(output_path, "w") as f:
        yaml.dump(params, f, default_flow_style=False)


def run_parameter_sweep(
    method: str,
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    data_path: Path = Path("data/pidnebesna"),
    dry_run: bool = True,
) -> None:
    """
    Run a parameter sweep for a specific method.

    Args:
        method: Method name (e.g., "InteriorPoint")
        base_config: Base configuration dict
        param_grid: Dictionary of parameter names to lists of values to try
        data_path: Path to data directory
        dry_run: If True, only show what would be run
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    print(f"Parameter sweep for {method}:")
    print(f"Parameters: {param_names}")
    print(f"Total combinations: {len(list(itertools.product(*param_values)))}")
    print()

    for combination in itertools.product(*param_values):
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

        # Create temporary config file
        config_file = Path(f"config_sweep_{method}.yaml")
        create_config_file(config, config_file)

        # Build command
        cmd = [
            "python",
            "run.py",
            "--methods",
            method,
            "--config",
            str(config_file),
            "--data-path",
            str(data_path),
        ]

        # Show what would be run
        print(f"Configuration: {dict(zip(param_names, combination))}")

        if dry_run:
            print(f"Would run: {' '.join(cmd)}")
        else:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)

        print()

        # Clean up config file
        if not dry_run:
            config_file.unlink()


# Example usage
if __name__ == "__main__":
    # Base configuration (shared across all runs)
    base_config = {"num_runs": 5, "num_workers": 4}

    # Example 1: Parameter sweep for InteriorPoint
    print("=" * 60)
    print("Example 1: InteriorPoint parameter sweep")
    print("=" * 60)

    #c_space = list(map(float, (np.linspace(0, 0.5, 11)[1:])))
    c_space = [0.7, 0.85, 1] #list(map(float, (np.linspace(0, 1, 21)[1:])))
    param_grid_interior = {
        "c": c_space,
        "max_iter": [500],
    }

    run_parameter_sweep(
        method="InteriorPoint",
        base_config=base_config,
        param_grid=param_grid_interior,
        dry_run=False,  # Set to False to actually run
    )

    sys.exit()

    # Example 2: Parameter sweep for OT4P4AS
    print("\n" + "=" * 60)
    print("Example 2: OT4P4AS parameter sweep")
    print("=" * 60)

    param_grid_ot4p = {
        "c": [0.1, 0.5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "initial_tau": [0.5, 0.7, 0.9],
        "final_tau": [0.1, 0.3, 0.5],
    }

    run_parameter_sweep(
        method="OT4P4AS",
        base_config=base_config,
        param_grid=param_grid_ot4p,
        dry_run=True,  # Set to False to actually run
    )

    # Example 3: Quick comparison sweep
    print("\n" + "=" * 60)
    print("Example 3: Quick comparison across methods")
    print("=" * 60)

    methods_to_compare = ["InteriorPoint", "QSA", "Manifold"]
    c_values = [0.01, 0.1, 1.0]

    for method in methods_to_compare:
        for c in c_values:
            config = base_config.copy()
            config["c"] = c

            # Use default method parameters
            config_file = Path(f"config_{method}_c{c}.yaml")
            create_config_file(config, config_file)

            cmd = [
                "python",
                "run.py",
                "--methods",
                method,
                "--config",
                str(config_file),
                "--instances",
                "instance1",  # Run on specific instance
                "--simulations",
                "0-4",  # Only first 5 simulations
            ]

            print(f"{method} with c={c}: {' '.join(cmd)}")
