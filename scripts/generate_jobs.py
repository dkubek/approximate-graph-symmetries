#!/usr/bin/env python3
# generate_jobs.py - Enhanced version with method-specific PBS generation

import os
import yaml
import json
import itertools
from pathlib import Path
import sys
import argparse
from jinja2 import Environment, FileSystemLoader, select_autoescape


# Graph type specific simulation counts
GRAPH_TYPE_SIMULATIONS = {
    "BA": 39,
    "ER": 39,
    "DD": 39,
    "GEO": 39,
    "GEO3D": 39,
    "HK": 39,
    "LRM_ER_rewired": 39,
    "BRAIN": 88,
}


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
    jinja_env,
    method_name,
    resources,
    job_range,
    output_dir,
    storage_path,
    project_dir,
    container_name,
    data_subdir,
    results_subdir,
    method_jobs,
):
    """Generate method-specific PBS script from Jinja2 template."""
    template = jinja_env.get_template("pbs_template.j2")

    # Simple job array name: AGS_<METHOD>
    # Configuration details can be found in configs/<method>/*.yaml using job array index
    job_array_name = f"AGS_{method_name}"

    # Prepare template variables
    template_vars = {
        "method_name": method_name,
        "job_array_name": job_array_name,
        "ncpus": resources["ncpus"],
        "mem": resources["mem"],
        "scratch_local": resources["scratch_local"],
        "walltime": resources["walltime"],
        "start_index": job_range[0],
        "end_index": job_range[1],
        "logs_dir": f"{output_dir}/logs",
        "sweep_dir": output_dir.name,  # Just the directory name, not full path
        "storage_path": storage_path,
        "project_dir": project_dir,
        "container_name": container_name,
        "data_subdir": data_subdir,
        "results_subdir": results_subdir,
    }

    # Render template
    pbs_content = template.render(**template_vars)  # Save PBS script
    pbs_file = output_dir / "pbs_scripts" / f"run_{method_name.lower()}_sweep.pbs"
    pbs_file.parent.mkdir(parents=True, exist_ok=True)

    with open(pbs_file, "w") as f:
        f.write(pbs_content)

    # Make executable
    pbs_file.chmod(0o755)

    return pbs_file


def create_job_configs(
    param_config_file,
    resource_config_file,
    template_dir,
    output_dir,
    storage_path,
    project_dir,
    container_name,
    data_subdir,
    results_subdir,
    graph_types=None,
    split_by_graph_type=True,
):
    """Create all job configuration files and PBS scripts."""
    param_config = load_parameter_config(param_config_file)
    resource_config = load_resource_config(resource_config_file)

    # Define available graph types
    available_graph_types = ["BA", "DD", "ER", "GEO", "GEO3D", "HK", "LRM_ER_rewired", "BRAIN"]

    # Determine which graph types to process
    if graph_types:
        graph_types_to_process = [
            gt for gt in graph_types if gt in available_graph_types
        ]
        if not graph_types_to_process:
            raise ValueError(
                f"No valid graph types specified. Available: {available_graph_types}"
            )
        print(f"Processing graph types: {graph_types_to_process}")
    else:
        graph_types_to_process = available_graph_types
        print(f"Processing all graph types: {graph_types_to_process}")

    # Set up Jinja2 environment
    jinja_env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
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

        # Create config files
        method_dir = output_dir / "configs" / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        for combo in combinations:
            # Determine graph type iteration strategy
            if split_by_graph_type:
                # Create separate job for each graph type
                graph_type_list = [[gt] for gt in graph_types_to_process]
            else:
                # Single job processes all graph types
                graph_type_list = [graph_types_to_process]
            
            for graph_types_subset in graph_type_list:
                # Determine simulation count based on graph type(s)
                if len(graph_types_subset) == 1:
                    # Single graph type - use its specific simulation count
                    total_simulations = GRAPH_TYPE_SIMULATIONS.get(
                        graph_types_subset[0], 39
                    )
                else:
                    # Multiple graph types - use default (should rarely happen)
                    total_simulations = 39
                
                # Get simulation splitting configuration
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
                    sim_ranges = [f"0-{total_simulations - 1}"]

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
                        "graph_types": graph_types_subset,  # Add graph types to config
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
                        "config_file": str(
                            config_file.relative_to(output_dir / "configs")
                        ),
                        "params": combo,
                        "simulation_range": sim_range,
                        "graph_types": ",".join(
                            graph_types_subset
                        ),  # Store as comma-separated string
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
                f.write(
                    f"{job['job_id']} {job['method']} {job['config_file']} {job['simulation_range']} {job['graph_types']}\n"
                )

        # Generate PBS script for this method
        if method_name not in resource_config["method_resources"]:
            print(
                f"WARNING: No resource configuration for {method_name}, using QSA defaults"
            )
            resources = resource_config["method_resources"]["QSA"]
        else:
            resources = resource_config["method_resources"][method_name]

        pbs_file = generate_pbs_script(
            jinja_env,
            method_name,
            resources,
            (method_start_idx, method_end_idx),
            output_dir,
            storage_path,
            project_dir,
            container_name,
            data_subdir,
            results_subdir,
            method_jobs,
        )

        # Store relative path to PBS script for submission script
        pbs_relative = pbs_file.relative_to(output_dir)
        submission_commands.append(
            {
                "method": method_name,
                "pbs_script": str(pbs_relative),
            }
        )

        print(f"\n{method_name}:")
        print(f"  Jobs: {len(method_jobs)} (array indices 1-{len(method_jobs)})")
        print(
            f"  Resources: {resources['ncpus']} CPUs, {resources['mem']} RAM, {resources['walltime']} walltime"
        )
        print(f"  PBS script: {pbs_file}")

    # Save master job list
    with open(output_dir / "all_jobs.json", "w") as f:
        json.dump(all_jobs, f, indent=2)

    # Create submission script using Jinja2 template
    submit_script = output_dir / "submit_all_methods.sh"
    template = jinja_env.get_template("submit_all_methods.j2")

    submit_content = template.render(jobs=submission_commands)

    with open(submit_script, "w") as f:
        f.write(submit_content)

    submit_script.chmod(0o755)

    # Create rerun utility script using Jinja2 template
    rerun_script = output_dir / "rerun_job.sh"
    template = jinja_env.get_template("rerun_job_template.j2")

    # Get list of methods that have jobs
    methods_with_jobs = list(jobs_by_method.keys())

    rerun_content = template.render(
        methods=methods_with_jobs,
        sweep_dir=output_dir.name,
    )

    with open(rerun_script, "w") as f:
        f.write(rerun_content)

    rerun_script.chmod(0o755)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Total jobs generated: {len(all_jobs)}")
    print(f"Output directory: {output_dir}")
    print(f"\nTo submit all jobs, run:")
    print(f"  {submit_script}")
    print(f"\nTo rerun specific jobs:")
    print(f"  {rerun_script} <method> <indices>")
    print(f"  Examples: {rerun_script} QSA 5")
    print(f"            {rerun_script} Manifold 1-10")
    print(f"            {rerun_script} InteriorPoint 1,5,10-15")
    print(f"{'=' * 60}")

    return all_jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PBS job scripts for parameter sweeps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "param_config",
        help="Path to parameter sweep configuration YAML file",
    )
    parser.add_argument(
        "resource_config",
        help="Path to method resources configuration YAML file",
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for generated job files",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--storage-path",
        default="/storage/praha1/home/$USER",
        help="Storage path on PBS cluster",
    )
    parser.add_argument(
        "--project-dir",
        default="ags",
        help="Project directory name (relative to storage path)",
    )
    parser.add_argument(
        "--container-name",
        default="ags-optimizer.sif",
        help="Singularity container filename",
    )
    parser.add_argument(
        "--data-subdir",
        default="data/kubek",
        help="Data subdirectory (relative to project dir)",
    )
    parser.add_argument(
        "--results-subdir",
        default="results-paper",
        help="Results subdirectory (relative to project dir)",
    )
    parser.add_argument(
        "--graph-types",
        nargs="+",
        default=None,
        help="Graph types to process (e.g., BA ER HK). If not specified, all types are processed.",
    )
    parser.add_argument(
        "--no-split-by-graph-type",
        dest="split_by_graph_type",
        action="store_false",
        help="Do NOT create separate jobs for each graph type (default: split enabled)",
    )
    parser.set_defaults(split_by_graph_type=True)

    args = parser.parse_args()  # Use templates from same directory as this script
    script_dir = Path(__file__).parent

    # Check that required templates exist
    required_templates = ["pbs_template.j2", "submit_all_methods.j2"]
    for template in required_templates:
        template_path = script_dir / template
        if not template_path.exists():
            print(f"ERROR: Template not found at {template_path}")
            sys.exit(1)

    create_job_configs(
        args.param_config,
        args.resource_config,
        script_dir,
        args.output_dir,
        args.storage_path,
        args.project_dir,
        args.container_name,
        args.data_subdir,
        args.results_subdir,
        args.graph_types,
        args.split_by_graph_type,
    )
