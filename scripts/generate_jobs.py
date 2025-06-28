#!/usr/bin/env python3
# File: generate_jobs.py
import yaml
import json
import itertools
from pathlib import Path
import sys


def load_parameter_config(config_file):
    """Load parameter sweep configuration from YAML file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def generate_parameter_combinations(method_name, method_params):
    """Generate all parameter combinations for a method."""
    param_names = list(method_params.keys())
    param_values = list(method_params.values())

    combinations = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)

    return combinations


def create_job_configs(config_file, output_dir):
    """Create individual job configuration files for all parameter combinations."""
    config = load_parameter_config(config_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create jobs directory structure
    jobs_dir = output_dir / "jobs"
    configs_dir = output_dir / "configs"

    all_jobs = []
    job_id = 0

    # Generate combinations for each method
    for method_params in config["parameter_sweeps"]:
        method_name = method_params["method"]
        del method_params["method"]

        print(method_name)
        method_dir = configs_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        combinations = generate_parameter_combinations(method_name, method_params)

        for combo in combinations:
            job_id += 1

            # Create full config for this job
            job_config = {
                "job_id": job_id,
                "method": method_name,
                "c": combo.get("c", 0.1),
                "num_runs": config["global_params"]["num_runs"],
                "num_workers": config["global_params"]["num_workers"],
                method_name: {k: v for k, v in combo.items() if k != "c"},
            }

            # Save individual config file
            config_file = method_dir / f"job_{job_id:04d}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(job_config, f, default_flow_style=False)

            # Add to master list
            all_jobs.append(
                {
                    "job_id": job_id,
                    "method": method_name,
                    "config_file": str(config_file.relative_to(configs_dir)),
                    "params": combo,
                }
            )

    # Save master job list
    with open(output_dir / "all_jobs.json", "w") as f:
        json.dump(all_jobs, f, indent=2)

    # Create parameter file for PBS array job
    with open(output_dir / "job_parameters.txt", "w") as f:
        for job in all_jobs:
            f.write(f"{job['job_id']} {job['method']} {job['config_file']}\n")

    print(f"Generated {len(all_jobs)} job configurations")
    print(f"Output directory: {output_dir}")

    # Print summary by method
    method_counts = {}
    for job in all_jobs:
        method = job["method"]
        method_counts[method] = method_counts.get(method, 0) + 1

    print("\nJobs per method:")
    for method, count in method_counts.items():
        print(f"  {method}: {count}")

    return all_jobs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_jobs.py <parameter_config.yaml> <output_dir>")
        sys.exit(1)

    create_job_configs(sys.argv[1], sys.argv[2])
