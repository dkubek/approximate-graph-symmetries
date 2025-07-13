#!/bin/bash
# File: run_parameter_sweep.pbs
#
# This script runs a parameter sweep for the AGS project using a job array.
# Each job task copies its data to a local scratch directory, runs the
# experiment inside a Singularity container, and cleans up afterwards.

# --- PBS Directives ---
#PBS -N AGS_ParamSweep
#PBS -l select=1:ncpus=16:mem=8gb:scratch_local=8gb
#PBS -l walltime=4:00:00
#PBS -o /storage/praha1/home/kubekd/ags/sweep/logs/job_${PBS_ARRAY_INDEX}.out
#PBS -e /storage/praha1/home/kubekd/ags/sweep/logs/job_${PBS_ARRAY_INDEX}.err
#PBS -m ae

# --- Script Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -eo pipefail

# --- Pre-flight Checks ---
# Check that the script is being run by PBS and not directly
[[ -n "$SCRATCHDIR" ]] || {
    echo >&2 "FATAL: Variable SCRATCHDIR is not set. This script must be run via qsub."
    exit 1
}
[[ -n "$PBS_ARRAY_INDEX" ]] || {
    echo >&2 "FATAL: Variable PBS_ARRAY_INDEX is not set. This script must be submitted as a job array (e.g., qsub -J 1-100)."
    exit 1
}

# --- Path Definitions ---
STORAGE="/storage/praha1/home/kubekd"
PROJECT_DIR="$STORAGE/ags"
SWEEP_DIR="$PROJECT_DIR/sweep"
CONTAINER="$PROJECT_DIR/ags-optimizer.sif"
DATA_DIR="$PROJECT_DIR/data/pidnebesna"
RESULTS_BASE="$PROJECT_DIR/results"

# --- Local (Scratch) Paths ---
LOCAL_CONFIG_NAME="config.yaml"
LOCAL_DATA_DIR="data" # The trailing slash is not needed here

trap 'clean_scratch' TERM EXIT

# --- Main Functions ---

# Copies all necessary files to the scratch directory
# Uses rsync for better performance and verbose output
_setup() {
    echo "--- SETUP: Preparing scratch directory: $SCRATCHDIR ---"
    cd "$SCRATCHDIR" || {
        echo >&2 "FATAL: Could not change to scratch directory."
        exit 1
    }

    echo "Copying data files..."
    rsync -av "$DATA_DIR/" "$LOCAL_DATA_DIR/" || {
        echo >&2 "FATAL: rsync of data failed."
        exit 2
    }

    echo "Copying parameters file..."
    cp -v "$SWEEP_DIR/job_parameters.txt" . || {
        echo >&2 "FATAL: Could not copy job_parameters.txt."
        exit 2
    }

    # Create the results directory if it doesn't exist
    mkdir -p "$RESULTS_BASE"
}

# Runs the primary computation
_main() {
    echo "--- MAIN: Starting computation ---"
    # Get parameters for this specific array job
    # The || true prevents the script from exiting if the last line of the file is empty
    params=$(sed -n "${PBS_ARRAY_INDEX}p" job_parameters.txt || true)
    if [[ -z "$params" ]]; then
        echo >&2 "FATAL: No parameters found for array index ${PBS_ARRAY_INDEX} in job_parameters.txt"
        exit 3
    fi

    # Read parameters into variables
    read -r job_id method config_file <<<"$params"

    echo "Copying config file: $config_file"
    cp "$SWEEP_DIR/configs/$config_file" "$LOCAL_CONFIG_NAME" || {
        echo >&2 "FATAL: Could not copy config file: $config_file"
        exit 2
    }

    echo "Starting Job ID: $job_id"
    echo "Array Index:     ${PBS_ARRAY_INDEX}"
    echo "Method:          $method"
    echo "Config file:     $config_file"
    echo "Start Time:      $(date)"

    # Set up the environment for Singularity and Python
    export SINGULARITY_BINDPATH="$SCRATCHDIR,$RESULTS_BASE"
    export PYTHONUNBUFFERED=1

    # Execute the Python script inside the container
    singularity exec "$CONTAINER" python3.11 /app/run.py \
        --methods "$method" \
        --config "$LOCAL_CONFIG_NAME" \
        --data-path "./$LOCAL_DATA_DIR" \
        --results-path "$RESULTS_BASE"

    # Capture the exit status of the singularity command
    local exit_status=$?
    echo "End Time:        $(date)"
    echo "--- MAIN COMPLETED with exit status: $exit_status ---"
    return $exit_status
}

# --- Script Execution ---
_setup
_main
main_exit_status=$?
exit $main_exit_status
