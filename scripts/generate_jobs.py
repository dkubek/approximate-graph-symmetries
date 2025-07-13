#!/usr/bin/env python3
# generate_jobs.py - Enhanced version with method-specific PBS generation

import os
import yaml
import json
import itertools
from pathlib import Path
import sys
from string import Template


def load_parameter_config(config_file):
    """Load parameter sweep configuration from YAML file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def load_resource_config(config_file):
    """Load method resource requirements from YAML file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def generate_parameter_combinations(method_name, method_params):
    """Generate all parameter combinations for a method."""
    # Remove method name from params if present
    if "method" in method_params:
        del method_params["method"]

    param_names = list(method_params.keys())
    param_values = list(method_params.values())

    combinations = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)

    return combinations


def split_simulation_ranges(total_sims, group_size):
    """Split total simulations into ranges of group_size."""
    ranges = []
    for start in range(0, total_sims, group_size):
        end = min(start + group_size - 1, total_sims - 1)
        ranges.append(f"{start}-{end}")
    return ranges


def generate_pbs_script(
    template_file, method_name, resources, job_range, output_dir, storage_path
):
    """Generate method-specific PBS script from template."""
    with open(template_file, "r") as f:
        template_content = f.read()

    # Create substitutions
    substitutions = {
        "METHOD_NAME": method_name,
        "NCPUS": str(resources["ncpus"]),
        "MEM": resources["mem"],
        "SCRATCH": resources["scratch_local"],
        "WALLTIME": resources["walltime"],
        "START_INDEX": str(job_range[0]),
        "END_INDEX": str(job_range[1]),
        "LOGS_DIR": f"{output_dir}/logs",
        "SWEEP_DIR": f"{output_dir}",
        "STORAGE_PATH": storage_path,
    }

    # Replace all placeholders
    pbs_content = template_content
    for key, value in substitutions.items():
        pbs_content = pbs_content.replace(f"<{key}>", value)

    # Save PBS script
    pbs_file = output_dir / "pbs_scripts" / f"run_{method_name.lower()}_sweep.pbs"
    pbs_file.parent.mkdir(parents=True, exist_ok=True)

    with open(pbs_file, "w") as f:
        f.write(pbs_content)

    # Make executable
    pbs_file.chmod(0o755)

    return pbs_file


def create_job_configs(
    param_config_file, resource_config_file, template_file, output_dir, storage_path
):
    """Create job configurations organized by method."""
    param_config = load_parameter_config(param_config_file)
    resource_config = load_resource_config(resource_config_file)
    output_dir = Path(output_dir)

    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "pbs_scripts").mkdir(exist_ok=True)

    # Track jobs by method
    jobs_by_method = {}
    all_jobs = []
    global_job_id = 0

    # Process each method configuration
    for method_spec in param_config["parameter_sweeps"]:
        method_name = method_spec["method"]

        if method_name not in jobs_by_method:
            jobs_by_method[method_name] = []

        # Generate combinations
        combinations = generate_parameter_combinations(method_name, method_spec.copy())

        # Get simulation splitting configuration
        total_simulations = 39  # Adjust this as needed or make it configurable
        split_sims = None
        if method_name in resource_config["method_resources"]:
            split_sims = resource_config["method_resources"][method_name].get(
                "split_simulations"
            )

        # Generate simulation ranges
        if split_sims:
            sim_ranges = split_simulation_ranges(total_simulations, split_sims)
        else:
            # No splitting - single range covering all simulations
            sim_ranges = [f"0-{total_simulations-1}"]

        # Create config files
        method_dir = output_dir / "configs" / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        for combo in combinations:
            for sim_range in sim_ranges:
                global_job_id += 1

                # Create job config
                job_config = {
                    "job_id": global_job_id,
                    "method": method_name,
                    "c": combo.get("c", 0.1),
                    "num_runs": param_config["global_params"]["num_runs"],
                    "num_workers": param_config["global_params"]["num_workers"],
                    "simulation_range": sim_range,
                    method_name: {k: v for k, v in combo.items() if k != "c"},
                }

                # Save config file
                config_file = method_dir / f"job_{global_job_id:04d}.yaml"
                with open(config_file, "w") as f:
                    yaml.dump(job_config, f, default_flow_style=False)

                # Track job
                job_info = {
                    "job_id": global_job_id,
                    "method": method_name,
                    "config_file": str(config_file.relative_to(output_dir / "configs")),
                    "params": combo,
                    "simulation_range": sim_range,
                }

                jobs_by_method[method_name].append(job_info)
                all_jobs.append(job_info)

    # Create method-specific parameter files and PBS scripts
    submission_commands = []

    for method_name, method_jobs in jobs_by_method.items():
        if not method_jobs:
            continue

        # Create parameter file for this method
        params_file = output_dir / f"{method_name}_parameters.txt"
        method_start_idx = 1
        method_end_idx = len(method_jobs)

        with open(params_file, "w") as f:
            for idx, job in enumerate(method_jobs, 1):
                f.write(f"{job['job_id']} {job['method']} {job['config_file']} {job['simulation_range']}\n")

        # Generate PBS script for this method
        if method_name not in resource_config["method_resources"]:
            print(
                f"WARNING: No resource configuration for {method_name}, using QSA defaults"
            )
            resources = resource_config["method_resources"]["QSA"]
        else:
            resources = resource_config["method_resources"][method_name]

        pbs_file = generate_pbs_script(
            template_file,
            method_name,
            resources,
            (method_start_idx, method_end_idx),
            output_dir,
            storage_path,
        )

        submission_commands.append(f"qsub {pbs_file}")

        print(f"\n{method_name}:")
        print(f"  Jobs: {len(method_jobs)} (array indices 1-{len(method_jobs)})")
        print(
            f"  Resources: {resources['ncpus']} CPUs, {resources['mem']} RAM, {resources['walltime']} walltime"
        )
        print(f"  PBS script: {pbs_file}")

    # Save master job list
    with open(output_dir / "all_jobs.json", "w") as f:
        json.dump(all_jobs, f, indent=2)

    # Create submission script
    submit_script = output_dir / "submit_all_methods.sh"
    with open(submit_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all method-specific parameter sweeps\n\n")
        f.write("echo 'Submitting parameter sweeps...'\n\n")

        for cmd in submission_commands:
            f.write(f"echo 'Submitting: {cmd}'\n")
            f.write(f"{cmd}\n")
            f.write("sleep 1\n\n")

        f.write("echo 'All jobs submitted!'\n")
        f.write("echo 'Monitor with: qstat -u $USER'\n")

    submit_script.chmod(0o755)

    # Summary
    print(f"\n{'='*60}")
    print(f"Total jobs generated: {len(all_jobs)}")
    print(f"Output directory: {output_dir}")
    print(f"\nTo submit all jobs, run:")
    print(f"  {submit_script}")
    print(f"{'='*60}")

    return all_jobs


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python generate_jobs.py <parameter_config.yaml> <method_resources.yaml> <output_dir> [storage_path]"
        )
        sys.exit(1)

    param_config = sys.argv[1]
    resource_config = sys.argv[2]
    output_dir = sys.argv[3]
    storage_path = sys.argv[4] if len(sys.argv) > 4 else "/storage/praha1/home/$USER"

    # Use the template from same directory as this script
    script_dir = Path(__file__).parent
    template_file = script_dir / "pbs_template.txt"

    if not template_file.exists():
        print(f"ERROR: PBS template not found at {template_file}")
        sys.exit(1)

    create_job_configs(
        param_config, resource_config, template_file, output_dir, storage_path
    )
