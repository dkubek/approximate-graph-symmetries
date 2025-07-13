# Dockerfile
# Multi-stage Docker build with uv package manager
# Following official uv Docker integration guide

# Stage 1: Build stage with uv
FROM python:3.11-slim-bookworm AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    gfortran \
    git \
    libopenblas-dev \
    liblapack-dev \
    coinor-libipopt-dev \
    libmumps-dev \
    libmumps-seq-dev \
    libscotch-dev \
    libmetis-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first (for better caching)
COPY pyproject_container.toml pyproject.toml
#COPY uv.lock* .

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN uv pip install --no-cache -r pyproject.toml

# Install torch-linear-assignment from git (if not in pyproject.toml)
RUN uv pip install --no-cache git+https://github.com/ivan-chai/torch-linear-assignment.git

# Stage 2: Runtime stage
FROM python:3.11-slim-bookworm AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    libgfortran5 \
    coinor-libipopt1v5 \
    libmumps-5.5 \
    libmumps-seq-5.5 \
    libscotch-7.0 \
    libmetis5 \
    vim \
    nano \
    htop \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser AGS ./AGS
COPY --chown=appuser:appuser run.py .
COPY --chown=appuser:appuser config/config.yaml .

# Switch to non-root user
USER appuser

# Set entrypoint
ENTRYPOINT ["python3"]
CMD ["--version"]