# Approximate Graph Symmetries using Continuous Optimization

> This repository contains the source code and implementation for the Master's thesis, "Using relaxation to compute
> approximate symmetry of graphs," submitted to the Faculty of Mathematics and Physics at Charles University.

## Introduction

This project explores continuous optimization paradigms for solving the Approximate Symmetry Problem (ASP) in
networks. Traditional graph symmetry, defined by exact automorphisms, is often too rigid for real-world networks which
exhibit imperfections and noise. The ASP reframes the question from *if* a network is symmetric to *how* symmetric it is
by finding a non-trivial permutation of its vertices that minimizes structural disagreements.

This work implements and compares several distinct algorithmic approaches to the relaxed version of the ASP (rASP),
where the discrete search space of permutation matrices is relaxed into a continuous domain. The goal is to evaluate the
performance, efficiency, and robustness of these methods to establish a clear hierarchy and provide insights for future
research.

## Implemented Methods

The core of this repository is the implementation of five distinct optimization methods for the rASP:

1.  **Quadratic Symmetry Approximator (QSA):** The baseline first-order method using the Frank-Wolfe algorithm on the Birkhoff polytope.
2.  **Interior-Point Method (IP):** A second-order method that leverages curvature information using the IPOPT solver.
3.  **Manifold Optimization (MF):** A geometry-aware, first-order method that treats the search space as a Riemannian manifold.
4.  **Orthogonal Relaxation (OR):** An alternative relaxation approach that optimizes over the non-convex manifold of orthogonal matrices using the OT4P framework.
5.  **Dimensionality Reduction (DR):** A method that drastically reduces the parameter space from `O(n^2)` to `O(n)` by using the differentiable `SoftSort` operator.

## Getting Started

Follow these instructions to set up the environment and run the experiments locally.

### Prerequisites

Before installing the Python dependencies, you must have the **IPOPT library** installed on your system. 

**On Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install coinor-libipopt-dev
```

**On Fedora/CentOS:**
```bash
sudo dnf install coin-or-Ipopt-devel
```

You will also need **Python 3.11** or newer.

### Installation with `uv`

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dkubek/approximate-graph-symmetries.git
    cd approximate-graph-symmetries
    ```

2.  **Install `uv` (if you don't have it):**
    `uv` can be installed via `pip` or by using its standalone installer.
    ```bash
    # Recommended: Standalone installer
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    Follow the instructions to add `uv` to your shell's `PATH`.

3.  **Create a virtual environment and install dependencies:**
    This single command creates a virtual environment in a `.venv` directory and installs all required packages from `pyproject.toml`.
    ```bash
    uv pip install .
    ```

The environment is now ready.

## Usage

A helper tool for running the experiments is the `run.py` script. One can also view the `demo.ipynb` notebook.

### Running a Single Experiment

You can run a specific optimization method on a subset of the data. For example, to run the Interior-Point method on the
first 3 simulations of all available instances:

```bash
python run.py \
    --methods QSA \
    --simulations 0-2 \
    --config config/config.yaml \
    --data-path data/example \
    --results-path results
```

- `--methods`: Specify one or more methods to run (`QSA`, `InteriorPoint`, `Manifold`, `OrthogonalRelaxation`, `DimensionalityReduction`, or `all`).
- `--simulations`: Define a range or list of simulation indices (e.g., `0-4`, `5,10,15`).
- `--config`: Path to a YAML configuration file.
- `--data-path` and `--results-path`: Specify directories for input data and output results.

## Running with Docker

For maximum reproducibility, a Docker environment is provided. This ensures that all system-level and Python
dependencies are identical to the ones used for development.

The image is available at [ags-optimizer:latest](https://hub.docker.com/repository/docker/dkubek/ags-optimizer/general)

### Building the Docker Image

From the root of the repository, run:
```bash
docker build -t ags-optimizer .
```

### Running an Experiment in Docker

To run an experiment, you need to mount the local `data` and `results` directories into the container.

This command runs the same experiment as the local example above, but inside the Docker container:

```bash
docker run --rm \
    -v "$PWD/data:/app/data" \
    -v "$PWD/results:/app/results" \
    ags-optimizer \
    run.py \
        --methods QSA \
        --simulations 0-2 \
        --data-path data/example \
        --results-path results
```
- `-v "$PWD/data:/app/data"`: Mounts your local data directory into the container.
- `-v "$PWD/results:/app/results"`: Mounts your local results directory so that output files are saved to your host machine.
- `--rm`: Automatically removes the container when it exits.

## Project Structure

```
.
├── AGS/                  # Main source code for the optimization methods
│   ├── methods/
│   │   ├── interior_point/
│   │   ├── manifold/
│   │   ├── orthogonal_relaxation/
│   │   ├── qsa/
│   │   └── dimensionality_reduction/
│   ├── annealing.py
│   ├── initialization.py
│   ├── projection.py
│   └── ...
├── config/               # YAML configuration files for experiments
├── scripts/              # Scripts for running parameter sweeps and generating jobs
├── Dockerfile            # Docker configuration for reproducible environment
├── pyproject.toml        # Project metadata and dependencies for uv/pip
├── run.py                # Main script for running experiments
└── README.md             # This file
```

## Acknowledgments

This work was conducted as part of a Master's thesis at the **Institute of Computer Science, Charles University**.

Special thanks to my supervisor, **doc. Ing. et Ing. David Hartman, Ph.D. et Ph.D.**, for his guidance and support.

This project utilizes or adapts code from the following external sources:
-   **OT4P:** The `AGS/methods/orthogonal_relaxation/OT4P/` directory contains code from the official implementation of the OT4P paper. [Original Repository](https://github.com/YamingGuo98/OT4P)
-   **Pymanopt & Manopt:** The implementation of the Doubly Stochastic manifold in `AGS/methods/manifold/doublystochastic.py` was adapted from the `pymanopt` (Python) and `manopt` (MATLAB) toolboxes.

## License

The source code in this repository is licensed under the MIT License. See the `LICENSE` file for more details. Note that
third-party code in the `OT4P/` directory is subject to its original license.