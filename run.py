#!/usr/bin/env python3
"""
Graph Optimization Runner
Processes graph instances using various optimization methods with parallel execution.
"""

import argparse
import logging
import os
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


# ============================================================================
# Configuration Classes
# ============================================================================
@dataclass
class BaseConfig:
    """Base configuration for all methods."""

    max_iter: int = 500
    verbose: int = 2  # Always 0 to avoid interfering with progress bars


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

        # Check if already processed
        results_path = Path("results") / method_name / graph_type
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
        times = []

        for run_idx in range(config.num_runs):
            perm, P_mat, exec_time = process_single_run(A, method, config.c)
            permutations.append(perm)
            matrices.append(P_mat)
            times.append(exec_time)

        return {
            "method": method_name,
            "graph_type": graph_type,
            "instance_id": instance_id,
            "permutations": np.column_stack(permutations),
            "matrices": matrices,
            "times": times,
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

    # Create directories
    base_path = Path("results") / method_name / graph_type
    (base_path / "Permutations").mkdir(parents=True, exist_ok=True)
    (base_path / "RawOutputs").mkdir(parents=True, exist_ok=True)
    (base_path / "Times").mkdir(parents=True, exist_ok=True)

    # Save permutations
    perm_file = base_path / "Permutations" / f"{instance_id}.csv"
    np.savetxt(perm_file, result_data["permutations"], delimiter=",", fmt="%d")

    # Save raw matrices
    raw_file = base_path / "RawOutputs" / f"{instance_id}.npz"
    matrix_dict = {f"P_{i}": mat for i, mat in enumerate(result_data["matrices"])}
    np.savez(raw_file, **matrix_dict)

    # Save times
    times_file = base_path / "Times" / f"{instance_id}.csv"
    np.savetxt(times_file, result_data["times"], delimiter=",", fmt="%.6f")


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

    if dry_run:
        print("\nDry run - no processing will be performed.")
        print("\nSample work items:")
        for item in full_work_items[:10]:
            print(f"  {item[0]} - {item[2].stem}_sim{item[3]}")
        if len(full_work_items) > 10:
            print(f"  ... and {len(full_work_items) - 10} more")
        return

    # Process with multiprocessing
    #num_workers = config.num_workers or cpu_count()
    num_workers = 1
    print(f"Using {num_workers} workers")

    # Process and save immediately
    processed = 0
    skipped = 0
    failed = 0

    # with Pool(num_workers) as pool:
    #    with tqdm(total=total_items, desc="Processing simulations") as pbar:
    #        for result in pool.imap_unordered(process_simulation, full_work_items):

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
