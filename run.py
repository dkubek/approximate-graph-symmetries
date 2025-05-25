#!/usr/bin/env python3
"""
Graph Optimization Runner
Processes graph instances using various optimization methods with parallel execution.
"""

import os
import sys
import argparse
import logging
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from multiprocessing import Pool, cpu_count
import traceback

import numpy as np
import torch
import yaml
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Import optimization methods
from AGS.methods import InteriorPoint, Manifold, OT4P4AS, QSA, SoftSort

# Suppress warnings
warnings.filterwarnings("ignore")


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
    #optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)


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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Process a single optimization run.

    Returns:
        permutation: Best permutation (n,)
        P_doubly_stochastic: Doubly stochastic matrix (n, n)
        time: Execution time
    """
    result = method.solve(A, c)
    P_doubly_stochastic = result["P"]
    execution_time = result["time"]

    # Project to permutation using linear assignment
    cost_matrix = -P_doubly_stochastic
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create permutation vector
    permutation = np.zeros(A.shape[0], dtype=int)
    permutation[row_indices] = col_indices

    return permutation, P_doubly_stochastic, execution_time


def process_instance(
    args: Tuple[str, str, Path, Config, int],
) -> Optional[Dict[str, Any]]:
    """
    Process a single instance (with all its simulations).

    Args:
        args: Tuple of (method_name, graph_type, instance_file, config, worker_id)

    Returns:
        Dictionary with results or None if failed
    """
    method_name, graph_type, instance_file, config, worker_id = args

    try:
        # Load data
        data = np.load(instance_file)
        instance_id_base = instance_file.stem

        # Create method instance
        method = create_method_instance(method_name, config)

        # Results storage
        all_results = {}

        # Process each simulation
        for sim_idx in range(39):  # 0 to 38
            instance_id = f"{instance_id_base}_sim{sim_idx}"

            # Check if already processed
            results_path = Path("results") / method_name / graph_type
            perm_file = results_path / "Permutations" / f"{instance_id}.csv"

            if perm_file.exists():
                continue

            # Get adjacency matrix
            A = data[str(sim_idx)].astype(np.float64)

            # Run multiple times
            permutations = []
            matrices = []
            times = []

            for run_idx in range(config.num_runs):
                perm, P_mat, exec_time = process_single_run(A, method, config.c)
                permutations.append(perm)
                matrices.append(P_mat)
                times.append(exec_time)

            all_results[instance_id] = {
                "permutations": np.column_stack(permutations),
                "matrices": matrices,
                "times": times,
            }

        return {
            "method": method_name,
            "graph_type": graph_type,
            "instance_base": instance_id_base,
            "results": all_results,
        }

    except Exception as e:
        logging.error(f"Failed to process {instance_file}: {str(e)}")
        logging.debug(traceback.format_exc())
        raise e
        return None


def save_results(result_data: Dict[str, Any]) -> None:
    """Save results to appropriate files."""
    if not result_data:
        return

    method_name = result_data["method"]
    graph_type = result_data["graph_type"]

    # Create directories
    base_path = Path("results") / method_name / graph_type
    (base_path / "Permutations").mkdir(parents=True, exist_ok=True)
    (base_path / "RawOutputs").mkdir(parents=True, exist_ok=True)
    (base_path / "Times").mkdir(parents=True, exist_ok=True)

    # Save each instance
    for instance_id, data in result_data["results"].items():
        # Save permutations
        perm_file = base_path / "Permutations" / f"{instance_id}.csv"
        np.savetxt(perm_file, data["permutations"], delimiter=",", fmt="%d")

        # Save raw matrices
        raw_file = base_path / "RawOutputs" / f"{instance_id}.npz"
        matrix_dict = {f"P_{i}": mat for i, mat in enumerate(data["matrices"])}
        np.savez(raw_file, **matrix_dict)

        # Save times
        times_file = base_path / "Times" / f"{instance_id}.csv"
        np.savetxt(times_file, data["times"], delimiter=",", fmt="%.6f")


# ============================================================================
# Main Orchestration
# ============================================================================
def collect_instances(data_path: Path) -> list[tuple[str, Path]]:
    """Collect all instance files from data directory."""
    instances = []

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
                instances.append((graph_type, filepath))

    return sorted(instances)


def run_optimization(
    methods: list[str], data_path: Path, config: Config, dry_run: bool = False
) -> None:
    """Run optimization for specified methods."""

    # Collect all instances
    instances = collect_instances(data_path)

    if not instances:
        print("No instances found!")
        return

    # Create work items
    work_items = []
    for method_name in methods:
        for graph_type, instance_file in instances:
            work_items.append((method_name, graph_type, instance_file, config))

    total_items = len(work_items)
    print(f"\nFound {len(instances)} instance files")
    print(f"Running {len(methods)} method(s): {', '.join(methods)}")
    print(f"Total work items: {total_items}")

    if dry_run:
        print("\nDry run - no processing will be performed.")
        return

    # Process with multiprocessing
    num_workers = config.num_workers or cpu_count()
    print(f"Using {num_workers} workers")

    # Add worker IDs to work items
    work_items_with_id = [
        (w[0], w[1], w[2], w[3], i % num_workers) for i, w in enumerate(work_items)
    ]

    # Create progress bar
    with Pool(num_workers) as pool:
        with tqdm(total=total_items, desc="Processing instances") as pbar:
            for result in pool.imap_unordered(process_instance, work_items_with_id):
                if result:
                    save_results(result)
                pbar.update(1)

    print("\nProcessing complete!")


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

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually running",
    )
    parser.add_argument("--log-file", type=Path, help="Path to log file for errors")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO
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

    # Print configuration
    print("\nConfiguration:")
    print(f"  Penalty parameter (c): {config.c}")
    print(f"  Number of runs: {config.num_runs}")
    print(f"  Number of workers: {config.num_workers or 'auto'}")

    # Run optimization
    run_optimization(methods, args.data_path, config, args.dry_run)


if __name__ == "__main__":
    main()
