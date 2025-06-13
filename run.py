#!/usr/bin/env python3
"""
Approximate Symmetry Optimization Runner

Processes graph instances using various optimization methods with optional
parallel execution.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Import optimization methods
from AGS.methods import OT4P4AS, QSA, InteriorPoint, Manifold, SoftSort

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress Numba debug logging
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)


# ============================================================================
# Configuration Classes
# ============================================================================
@dataclass
class BaseConfig:
    """Base configuration for all methods."""

    max_iter: int = 500
    verbose: int = 0  # Always 0 to avoid interfering with progress bars


@dataclass
class InteriorPointConfig(BaseConfig):
    """Configuration for Interior Point method."""

    max_iter: int = 1000
    tol: float = 1e-8


@dataclass
class ManifoldConfig(BaseConfig):
    """Configuration for Manifold method."""

    optimizer: str = "steepest_descent"
    max_iter: int = 500
    # optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OT4PConfig(BaseConfig):
    """Configuration for OT4P method."""

    max_iter: int = 3000
    initial_tau: float = 0.7
    final_tau: float = 0.5
    annealing_scheme: str = "exponential"
    decay_steps: int = 5000
    learning_rate: float = 0.1
    min_rel_improvement: float = 1e-4
    verbose: int = 1


@dataclass
class QSAConfig(BaseConfig):
    """Configuration for QSA method."""

    max_iter: int = 500
    tol: float = 1e-8


@dataclass
class SoftSortConfig(BaseConfig):
    """Configuration for SoftSort method."""

    max_iter: int = 2000
    initial_tau: float = 1.0
    final_tau: float = 1e-6
    annealing_scheme: str = "cosine"
    learning_rate: float = 0.1
    min_rel_improvement: float = 1e-4


@dataclass
class Config:
    """Main configuration container."""

    c: float = 0.1  # Penalty parameter
    num_runs: int = 5  # Number of runs per instance
    num_workers: Optional[int] = None  # Number of parallel workers

    # Method configurations
    InteriorPoint: InteriorPointConfig = field(default_factory=InteriorPointConfig)
    Manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    OT4P4AS: OT4PConfig = field(default_factory=OT4PConfig)
    QSA: QSAConfig = field(default_factory=QSAConfig)
    SoftSort: SoftSortConfig = field(default_factory=SoftSortConfig)


# ============================================================================
# Parameter Directory Name Generation
# ============================================================================
def format_value_for_dirname(value: Any) -> str:
    """
    Format a parameter value for use in directory name.

    - Floats: decimal points → '_', scientific notation → 'em'
    - Strings: spaces → '_', lowercase
    - Others: string representation
    """
    if isinstance(value, float):
        # Check if scientific notation is more appropriate
        abs_val = abs(value)
        if abs_val != 0 and (abs_val < 0.001 or abs_val >= 10000):
            # Use scientific notation
            # Format with up to 6 significant figures
            exp_str = f"{value:.6e}"
            # Replace 'e-' with 'em' and 'e+' with 'ep'
            exp_str = exp_str.replace("e-", "em").replace("e+", "ep")
            # Remove trailing zeros after decimal
            if "." in exp_str:
                base, exp = exp_str.split("e" if "e" in exp_str else "E")
                base = base.rstrip("0").rstrip(".")
                exp_str = (
                    base
                    + ("em" if "em" in exp_str else "ep")
                    + exp.split("m" if "m" in exp else "p")[1]
                )
            return exp_str
        else:
            # Regular notation - format with up to 6 significant figures
            formatted = f"{value:.6g}"
            # Replace dots with underscores
            return formatted.replace(".", "_")
    elif isinstance(value, str):
        # Replace spaces and other problematic characters
        return value.lower().replace(" ", "_").replace("-", "_")
    else:
        return str(value)


def get_important_params(
    method_name: str, method_config: Any, c: float
) -> Dict[str, Any]:
    """
    Get important parameters for a given method.

    Returns a dictionary of parameter names and values that should be
    included in the directory name.
    """
    # Always include c
    params = {"c": c}

    # Method-specific important parameters
    if method_name == "InteriorPoint":
        params.update(
            {
                "max_iter": method_config.max_iter,
                "tol": method_config.tol,
            }
        )
    elif method_name == "Manifold":
        params.update(
            {
                "optimizer": method_config.optimizer,
                "max_iter": method_config.max_iter,
            }
        )
    elif method_name == "OT4P4AS":
        params.update(
            {
                "max_iter": method_config.max_iter,
                "initial_tau": method_config.initial_tau,
                "final_tau": method_config.final_tau,
                "learning_rate": method_config.learning_rate,
            }
        )
    elif method_name == "QSA":
        params.update(
            {
                "max_iter": method_config.max_iter,
                "tol": method_config.tol,
            }
        )
    elif method_name == "SoftSort":
        params.update(
            {
                "max_iter": method_config.max_iter,
                "initial_tau": method_config.initial_tau,
                "final_tau": method_config.final_tau,
                "learning_rate": method_config.learning_rate,
            }
        )

    return params


def get_param_dirname(method_name: str, method_config: Any, config: Config) -> str:
    """
    Generate parameter directory name for a given method and configuration.

    Example: "c0_1_max_iter1000_tol1em8"
    """
    # Get important parameters
    params = get_important_params(method_name, method_config, config.c)

    # Sort parameters alphabetically for consistency
    sorted_params = sorted(params.items())

    # Format each parameter
    param_strings = []
    for param_name, param_value in sorted_params:
        formatted_value = format_value_for_dirname(param_value)
        param_strings.append(f"{param_name}{formatted_value}")

    return "_".join(param_strings)


def save_config_json(
    base_path: Path, method_name: str, method_config: Any, config: Config
) -> None:
    """
    Save complete configuration to JSON file.

    Saves all parameters including c, num_runs, and method-specific parameters.
    """
    config_path = base_path / "config.json"

    # Build config dictionary
    config_dict = {
        "method": method_name,
        "c": config.c,
        "num_runs": config.num_runs,
        "method_config": asdict(method_config),
    }

    # Save as JSON with proper formatting
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def create_method_instance(method_name: str, config: Config) -> Any:
    """Create an instance of the specified method."""
    method_config = getattr(config, method_name)

    if method_name == "InteriorPoint":
        return InteriorPoint(**asdict(method_config))
    elif method_name == "Manifold":
        return Manifold(**asdict(method_config))
    elif method_name == "OT4P4AS":
        return OT4P4AS(**asdict(method_config))
    elif method_name == "QSA":
        return QSA(**asdict(method_config))
    elif method_name == "SoftSort":
        return SoftSort(**asdict(method_config))
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ============================================================================
# Processing Functions
# ============================================================================
def process_single_run(
    A: np.ndarray, method: Any, c: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Process a single optimization run.

    Returns:
        permutation: Best permutation (n,)
        P_doubly_stochastic: Doubly stochastic matrix (n, n)
        time: Execution time (deprecated, use metrics['time'])
        metrics: Dictionary of metrics from this run
    """
    result = method.solve(A, c)
    P_doubly_stochastic = result["P"]
    metrics = result["metrics"]

    # Project to permutation using linear assignment
    cost_matrix = -P_doubly_stochastic
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create permutation vector
    permutation = np.zeros(A.shape[0], dtype=int)
    permutation[row_indices] = col_indices

    return permutation, P_doubly_stochastic, metrics


def process_simulation(
    args: Tuple[str, str, Path, int, Config, bool],
) -> Optional[Dict[str, Any]]:
    """
    Process a single simulation.

    Args:
        args: Tuple of (method_name, graph_type, instance_file, sim_idx, config, force_recompute)

    Returns:
        Dictionary with results or None if failed/skipped
    """
    method_name, graph_type, instance_file, sim_idx, config, force_recompute = args

    try:
        instance_id_base = instance_file.stem
        instance_id = f"{instance_id_base}_sim{sim_idx}"

        # Get method config and parameter directory name
        method_config = getattr(config, method_name)
        param_dirname = get_param_dirname(method_name, method_config, config)

        # Check if already processed with new directory structure
        results_path = Path("results") / method_name / param_dirname / graph_type
        perm_file = results_path / "Permutations" / f"{instance_id}.csv"

        if perm_file.exists() and not force_recompute:
            return None  # Skip already processed

        # Load data
        data = np.load(instance_file)

        # Get adjacency matrix
        A = data[str(sim_idx)].astype(np.float64)

        # Create method instance
        method = create_method_instance(method_name, config)

        # Run multiple times
        permutations = []
        matrices = []
        all_metrics = []  # Collect metrics from all runs

        for run_idx in range(config.num_runs):
            perm, P_mat, metrics = process_single_run(A, method, config.c)
            permutations.append(perm)
            matrices.append(P_mat)
            all_metrics.append(metrics)

        return {
            "method": method_name,
            "graph_type": graph_type,
            "instance_id": instance_id,
            "param_dirname": param_dirname,
            "method_config": method_config,
            "config": config,
            "permutations": np.column_stack(permutations),
            "matrices": matrices,
            "metrics": all_metrics,
        }

    except Exception as e:
        logging.error(f"Failed to process {instance_file.stem}_sim{sim_idx}: {str(e)}")
        logging.debug(traceback.format_exc())
        return False


def save_single_result(result_data: Dict[str, Any]) -> None:
    """Save a single simulation result immediately."""
    if not result_data:
        return

    method_name = result_data["method"]
    graph_type = result_data["graph_type"]
    instance_id = result_data["instance_id"]
    param_dirname = result_data["param_dirname"]
    method_config = result_data["method_config"]
    config = result_data["config"]

    # Create directories with new structure
    base_path = Path("results") / method_name / param_dirname
    graph_path = base_path / graph_type
    (graph_path / "Permutations").mkdir(parents=True, exist_ok=True)
    (graph_path / "RawOutputs").mkdir(parents=True, exist_ok=True)
    (graph_path / "Metrics").mkdir(parents=True, exist_ok=True)

    # Save config.json if it doesn't exist
    if not (base_path / "config.json").exists():
        save_config_json(base_path, method_name, method_config, config)

    # Save permutations
    perm_file = graph_path / "Permutations" / f"{instance_id}.csv"
    np.savetxt(perm_file, result_data["permutations"], delimiter=",", fmt="%d")

    # Save raw matrices
    raw_file = graph_path / "RawOutputs" / f"{instance_id}.npz"
    matrix_dict = {f"P_{i}": mat for i, mat in enumerate(result_data["matrices"])}
    np.savez(raw_file, **matrix_dict)

    # Save metrics
    metrics_file = graph_path / "Metrics" / f"{instance_id}.csv"
    
    # Prepare metrics data for CSV
    metrics_list = result_data["metrics"]
    
    # Add run_id to each metric dict
    for i, metric_dict in enumerate(metrics_list):
        metric_dict["run_id"] = i
    
    # Write metrics to CSV
    if metrics_list:
        # Get all unique keys across all runs (for flexible columns)
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        # Sort keys for consistent column order (run_id first)
        fieldnames = ["run_id"] + sorted([k for k in all_keys if k != "run_id"])
        
        import csv
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_list)


# ============================================================================
# Main Orchestration
# ============================================================================
def collect_work_items(
    data_path: Path,
    instance_filter: Optional[List[str]] = None,
    simulation_filter: Optional[List[int]] = None,
) -> List[Tuple[str, Path, int]]:
    """
    Collect all work items (instance, simulation pairs) from data directory.

    Args:
        data_path: Path to data directory
        instance_filter: List of instance base names to process (None = all)
        simulation_filter: List of simulation indices to process (None = all)

    Returns:
        List of (graph_type, instance_file, sim_idx) tuples
    """
    work_items = []
    all_simulations = list(range(39))  # 0 to 38

    # Determine which simulations to process
    simulations_to_process = simulation_filter if simulation_filter else all_simulations

    for graph_type in os.listdir(data_path):
        graph_dir = data_path / graph_type
        if not graph_dir.is_dir():
            continue

        for filename in os.listdir(graph_dir):
            filepath = graph_dir / filename
            if (
                filepath.is_file()
                and filepath.suffix == ".npz"
                and "allInfo" not in filename
            ):
                instance_base = filepath.stem

                # Check instance filter
                if instance_filter and instance_base not in instance_filter:
                    continue

                # Add work items for each simulation
                for sim_idx in simulations_to_process:
                    work_items.append((graph_type, filepath, sim_idx))

    return sorted(work_items)


def run_optimization(
    methods: List[str],
    data_path: Path,
    config: Config,
    instance_filter: Optional[List[str]] = None,
    simulation_filter: Optional[List[int]] = None,
    force_recompute: bool = False,
    dry_run: bool = False,
) -> None:
    """Run optimization for specified methods."""

    # Collect all work items
    work_items = collect_work_items(data_path, instance_filter, simulation_filter)

    if not work_items:
        print("No work items found!")
        return

    # Create full work items with method
    full_work_items = []
    for method_name in methods:
        for graph_type, instance_file, sim_idx in work_items:
            full_work_items.append(
                (
                    method_name,
                    graph_type,
                    instance_file,
                    sim_idx,
                    config,
                    force_recompute,
                )
            )

    total_items = len(full_work_items)

    # Count unique instances and simulations
    unique_instances = len(set((item[1], item[2]) for item in work_items))
    # unique_simulations = len(set(item[3] for item in work_items))

    print(f"\nFound {unique_instances} instance files")
    # print(f"Processing {unique_simulations} simulation(s) per instance")
    print(f"Running {len(methods)} method(s): {', '.join(methods)}")
    print(f"Total work items: {total_items}")
    print(f"Force recompute: {'Yes' if force_recompute else 'No'}")

    # Display parameter configurations
    print("\nParameter configurations:")
    for method_name in methods:
        method_config = getattr(config, method_name)
        param_dirname = get_param_dirname(method_name, method_config, config)
        print(f"  {method_name}: {param_dirname}")

    if dry_run:
        print("\nDry run - no processing will be performed.")
        print("\nSample work items:")
        for item in full_work_items[:10]:
            print(f"  {item[0]} - {item[2].stem}_sim{item[3]}")
        if len(full_work_items) > 10:
            print(f"  ... and {len(full_work_items) - 10} more")
        return

    # Process with multiprocessing
    # num_workers = config.num_workers or cpu_count()
    num_workers = 1
    print(f"Using {num_workers} workers")

    # Process and save immediately
    processed = 0
    skipped = 0
    failed = 0

    # Disable multiprocessing
    # with Pool(num_workers) as pool:
    #     with tqdm(total=total_items, desc="Processing simulations") as pbar:
    #         for result in pool.imap_unordered(process_simulation, full_work_items):
    #             if result:
    #                 save_single_result(result)
    #                 processed += 1
    #             elif result is None:
    #                 skipped += 1
    #             else:
    #                 failed += 1
    #             pbar.update(1)
    #             pbar.set_postfix(processed=processed, skipped=skipped, failed=failed)

    with tqdm(total=total_items, desc="Processing simulations") as pbar:
        for work_item in full_work_items:
            result = process_simulation(work_item)
            if result:
                save_single_result(result)
                processed += 1
            elif result is None:
                skipped += 1
            else:
                failed += 1
            pbar.update(1)
            pbar.set_postfix(processed=processed, skipped=skipped, failed=failed)

    print(f"\nProcessing complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Failed: {failed}")


# ============================================================================
# Configuration Loading
# ============================================================================
def load_config(config_file: Optional[Path]) -> Config:
    """Load configuration from YAML file."""
    if config_file and config_file.exists():
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Create nested config objects
        config = Config()

        # Load top-level config
        for key in ["c", "num_runs", "num_workers"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        # Load method configs
        for method_name in ["InteriorPoint", "Manifold", "OT4P4AS", "QSA", "SoftSort"]:
            if method_name in config_dict:
                method_config_class = globals()[f"{method_name}Config"]
                method_config = method_config_class(**config_dict[method_name])
                setattr(config, method_name, method_config)

        return config
    else:
        return Config()


# ============================================================================
# CLI Interface
# ============================================================================
def parse_simulation_indices(sim_str: str) -> List[int]:
    """
    Parse simulation indices from string.
    Examples: "0,1,2" or "0-5" or "0-5,10,15-20"
    """
    simulations = []
    parts = sim_str.split(",")

    for part in parts:
        if "-" in part:
            start, end = map(int, part.split("-"))
            simulations.extend(range(start, end + 1))
        else:
            simulations.append(int(part))

    # Filter valid simulation indices (0-38)
    return [s for s in simulations if 0 <= s <= 38]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run graph optimization methods on instance files"
    )

    # Paths
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/pidnebesna"),
        help="Path to data directory (default: data/pidnebesna)",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results"),
        help="Path to results directory (default: results)",
    )

    # Method selection
    all_methods = ["InteriorPoint", "Manifold", "OT4P4AS", "QSA", "SoftSort"]
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=all_methods + ["all"],
        default=["all"],
        help="Methods to run (default: all)",
    )

    # Instance and simulation selection
    parser.add_argument(
        "--instances",
        nargs="+",
        help="Specific instance base names to process (e.g., instance1 instance2)",
    )
    parser.add_argument(
        "--simulations",
        type=str,
        help="Simulation indices to process (e.g., '0,1,2' or '0-5' or '0-5,10,15-20')",
    )

    # Configuration
    parser.add_argument("--config", type=Path, help="Path to YAML configuration file")
    parser.add_argument(
        "--c", type=float, help="Penalty parameter (overrides config file)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        help="Number of runs per instance (overrides config file)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of parallel workers (default: number of CPU cores)",
    )

    # Processing options
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of existing results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually running",
    )
    parser.add_argument("--log-file", type=Path, help="Path to log file for errors")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    if args.log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=log_level, format=log_format)

    # Load configuration
    config = load_config(args.config)

    # Override with CLI arguments
    if args.c is not None:
        config.c = args.c
    if args.num_runs is not None:
        config.num_runs = args.num_runs
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    # Determine methods to run
    if "all" in args.methods:
        methods = all_methods
    else:
        methods = args.methods

    # Parse simulation indices
    simulation_filter = None
    if args.simulations:
        simulation_filter = parse_simulation_indices(args.simulations)
        print(f"Processing simulations: {sorted(simulation_filter)}")

    # Print configuration
    print("\nConfiguration:")
    print(f"  Penalty parameter (c): {config.c}")
    print(f"  Number of runs: {config.num_runs}")
    print(f"  Number of workers: {config.num_workers or 'auto'}")

    # Run optimization
    run_optimization(
        methods,
        args.data_path,
        config,
        instance_filter=args.instances,
        simulation_filter=simulation_filter,
        force_recompute=args.force_recompute,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
