#!/usr/bin/env python3
"""
Example script showing how to perform parameter sweeps with parallel execution.
Supports both pure Python subprocess management and Unix xargs approach.
"""

import sys
import itertools
import subprocess
import yaml
import time
import os
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
    param_str = "_".join(f"{name}={value}" for name, value in zip(param_names, combination))
    # Replace problematic characters for filenames
    param_str = param_str.replace("/", "-").replace(" ", "_")
    return param_str


def run_command_with_logging(cmd: List[str], log_file: Path) -> subprocess.Popen:
    """Run a command with output redirected to a log file."""
    with open(log_file, "w") as f:
        return subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )


def run_parallel_python(
    commands: List[Tuple[List[str], Path, Path]],
    max_workers: int
) -> Dict[str, bool]:
    """
    Run commands in parallel using pure Python subprocess management.
    
    Returns a dictionary mapping config files to success status.
    """
    results = {}
    active_processes = {}
    command_queue = list(commands)
    
    print(f"\nRunning {len(commands)} combinations with {max_workers} parallel workers...")
    
    while command_queue or active_processes:
        # Start new processes if we have capacity
        while len(active_processes) < max_workers and command_queue:
            cmd, log_file, config_file = command_queue.pop(0)
            proc = run_command_with_logging(cmd, log_file)
            active_processes[proc] = (config_file, log_file)
        
        # Check for completed processes
        completed = []
        for proc in active_processes:
            if proc.poll() is not None:
                completed.append(proc)
        
        # Handle completed processes
        for proc in completed:
            config_file, log_file = active_processes.pop(proc)
            success = proc.returncode == 0
            results[str(config_file)] = success
            
            if not success:
                print(f"  ❌ Failed: {config_file.name} (see {log_file})")
            else:
                # Clean up config file on success
                config_file.unlink()
        
        # Brief sleep to avoid busy waiting
        if active_processes:
            time.sleep(0.1)
    
    return results


def run_parallel_xargs(
    commands: List[Tuple[List[str], Path, Path]],
    max_workers: int
) -> Dict[str, bool]:
    """
    Run commands in parallel using xargs.
    
    Returns a dictionary mapping config files to success status.
    """
    print(f"\nRunning {len(commands)} combinations using xargs with {max_workers} workers...")
    
    # Create a temporary script for xargs to execute
    script_path = Path("run_single_sweep.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to run a single parameter combination\n")
        f.write("CMD=\"$1\"\n")
        f.write("LOG=\"$2\"\n")
        f.write("CONFIG=\"$3\"\n")
        f.write("eval \"$CMD\" > \"$LOG\" 2>&1\n")
        f.write("EXIT_CODE=$?\n")
        f.write("if [ $EXIT_CODE -eq 0 ]; then\n")
        f.write("    rm -f \"$CONFIG\"\n")
        f.write("else\n")
        f.write("    echo \"Failed: $CONFIG (exit code: $EXIT_CODE)\" >&2\n")
        f.write("fi\n")
        f.write("exit $EXIT_CODE\n")
    
    script_path.chmod(0o755)
    
    # Create input for xargs
    xargs_input = []
    for cmd, log_file, config_file in commands:
        cmd_str = " ".join(cmd)
        xargs_input.append(f"{cmd_str}\t{log_file}\t{config_file}")
    
    # Run xargs
    xargs_cmd = ["xargs", "-P", str(max_workers), "-L", "1", "-I", "{}", 
                 "bash", "-c", f"./{script_path} {{}}", "_"]
    
    proc = subprocess.Popen(
        xargs_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = proc.communicate(input="\n".join(xargs_input))
    
    # Clean up script
    script_path.unlink()
    
    # Parse results from stderr
    results = {}
    for cmd, log_file, config_file in commands:
        # Check if config file still exists (failure) or was deleted (success)
        if config_file.exists():
            results[str(config_file)] = False
            config_file.unlink()  # Clean up
        else:
            results[str(config_file)] = True
    
    if stderr:
        print(f"\nErrors:\n{stderr}")
    
    return results


def run_parameter_sweep(
    method: str,
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    data_path: Path = Path("data/pidnebesna"),
    parallel: int = 1,
    use_xargs: bool = False,
) -> None:
    """
    Run a parameter sweep for a specific method.

    Args:
        method: Method name (e.g., "InteriorPoint")
        base_config: Base configuration dict
        param_grid: Dictionary of parameter names to lists of values to try
        data_path: Path to data directory
        parallel: Number of parallel processes (1 = sequential)
        use_xargs: If True, use xargs instead of Python subprocess management
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"\nParameter sweep for {method}:")
    print(f"Parameters: {param_names}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Parallel workers: {parallel}")
    print(f"Method: {'xargs' if use_xargs else 'Python subprocess'}")
    
    # Create log directory
    log_dir = create_log_directory(method)
    print(f"Log directory: {log_dir}")
    
    # Prepare all commands
    commands = []
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
        
        # Create log file path
        log_file = log_dir / f"run_{i:03d}_{param_str}.log"
        
        commands.append((cmd, log_file, config_file))
    
    # Run commands in parallel
    if parallel == 1:
        # Sequential execution (original behavior)
        print("\nRunning sequentially...")
        results = {}
        for cmd, log_file, config_file in commands:
            print(f"  Running: {config_file.name}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            
            # Write output to log file
            with open(log_file, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Exit code: {proc.returncode}\n")
                f.write("\n--- STDOUT ---\n")
                f.write(proc.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(proc.stderr)
            
            results[str(config_file)] = proc.returncode == 0
            
            if proc.returncode != 0:
                print(f"    ❌ Failed (see {log_file})")
            
            # Clean up config file
            config_file.unlink()
    else:
        # Parallel execution
        if use_xargs:
            results = run_parallel_xargs(commands, parallel)
        else:
            results = run_parallel_python(commands, parallel)
    
    # Summary
    elapsed = time.time() - start_time
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"Sweep completed in {elapsed:.1f} seconds")
    print(f"Successful runs: {successful}/{len(results)}")
    if failed > 0:
        print(f"Failed runs: {failed} (check log files in {log_dir})")
    print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Base configuration (shared across all runs)
    base_config = {"num_runs": 5, "num_workers": 10}
    
    # Example 1: Parameter sweep for Manifold with parallel execution
    #print("=" * 60)
    #print("Example 1: Manifold parameter sweep (parallel)")
    #print("=" * 60)
    
    #param_grid_manifold = {
    #    "c": [0.01, 0.1, 0.3, 0.5, 1.0],
    #    "max_iter": [100, 500, 2500],
    #}
    
    ## Run with 4 parallel workers using Python subprocess
    #run_parameter_sweep(
    #    method="Manifold",
    #    base_config=base_config,
    #    param_grid=param_grid_manifold,
    #    use_xargs=False,  # Use Python subprocess management
    #)

    param_grid = {
        "c": [0.001, 0.1, 0.5],
        "initial_tau": [0.3, 0.5, 0.7],
        "final_tau": [None],
    }

    run_parameter_sweep(
        method="OT4P4AS",
        base_config=base_config,
        param_grid=param_grid,
        parallel=9,  # Run 4 processes in parallel
        use_xargs=False,  # Use Python subprocess management
    )

    sys.exit()
    
    # Example 2: Same sweep using xargs
    print("\n" + "=" * 60)
    print("Example 2: Same sweep using xargs")
    print("=" * 60)
    
    run_parameter_sweep(
        method="Manifold",
        base_config=base_config,
        param_grid=param_grid_manifold,
        parallel=4,
        use_xargs=True,  # Use xargs for parallelism
    )
    
    # Example 3: Quick comparison with limited parallelism
    print("\n" + "=" * 60)
    print("Example 3: Quick comparison with 2 parallel workers")
    print("=" * 60)
    
    methods_to_compare = ["InteriorPoint", "QSA", "Manifold"]
    c_values = [0.01, 0.1, 1.0]
    
    for method in methods_to_compare:
        param_grid = {"c": c_values}
        run_parameter_sweep(
            method=method,
            base_config=base_config,
            param_grid=param_grid,
            parallel=2,  # Only 2 parallel workers
        )
