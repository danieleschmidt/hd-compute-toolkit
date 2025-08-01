# HD-Compute-Toolkit Multi-stage Dockerfile
# Optimized for development, testing, and production deployment

# =============================================================================
# Base stage with CUDA support
# =============================================================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip and install base Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 hdcuser
USER hdcuser
WORKDIR /home/hdcuser

# Set up Python virtual environment
RUN python3 -m venv /home/hdcuser/venv
ENV PATH="/home/hdcuser/venv/bin:$PATH"

# =============================================================================
# Development stage
# =============================================================================
FROM base as development

# Install development dependencies
COPY --chown=hdcuser:hdcuser requirements-dev.txt* ./
RUN if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi

# Install additional development tools
RUN pip install \
    jupyter \
    ipywidgets \
    jupyterlab \
    notebook \
    tensorboard \
    wandb \
    mlflow

# Copy source code
COPY --chown=hdcuser:hdcuser . /home/hdcuser/hd-compute-toolkit/
WORKDIR /home/hdcuser/hd-compute-toolkit

# Install package in development mode
RUN pip install -e ".[dev,docs]"

# Install pre-commit hooks
RUN pre-commit install || true

# Expose ports for development services
EXPOSE 8888 6006 5000

# Default command for development
CMD ["bash"]

# =============================================================================
# Testing stage  
# =============================================================================
FROM development as testing

# Install additional testing dependencies
RUN pip install \
    pytest-html \
    pytest-json-report \
    pytest-memprof \
    pytest-timeout

# Copy test configuration
COPY --chown=hdcuser:hdcuser pytest.ini ./
COPY --chown=hdcuser:hdcuser .coveragerc ./

# Set test environment variables
ENV PYTEST_CURRENT_TEST=1
ENV COVERAGE_CORE=sysmon

# Default command runs full test suite
CMD ["pytest", "tests/", "-v", "--cov=hd_compute", "--cov-report=html", "--cov-report=term-missing"]

# =============================================================================
# Production build stage
# =============================================================================
FROM base as builder

# Copy only necessary files for building
COPY --chown=hdcuser:hdcuser pyproject.toml ./
COPY --chown=hdcuser:hdcuser README.md ./
COPY --chown=hdcuser:hdcuser LICENSE ./
COPY --chown=hdcuser:hdcuser hd_compute/ ./hd_compute/

# Install build dependencies
RUN pip install build wheel

# Build the package
RUN python -m build

# =============================================================================
# Production runtime stage
# =============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libsndfile1 \
    libhdf5-103 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 hdcuser
USER hdcuser
WORKDIR /home/hdcuser

# Set up Python virtual environment
RUN python3 -m venv /home/hdcuser/venv
ENV PATH="/home/hdcuser/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Copy built package from builder stage
COPY --from=builder --chown=hdcuser:hdcuser /home/hdcuser/dist/*.whl ./

# Install the package
RUN pip install *.whl && rm *.whl

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hd_compute; print('OK')" || exit 1

# Default command
CMD ["python"]

# =============================================================================
# FPGA development stage (optional)
# =============================================================================
FROM development as fpga-dev

# Switch to root for FPGA tool installation
USER root

# Install FPGA development dependencies (Xilinx tools would be mounted)
RUN apt-get update && apt-get install -y \
    device-tree-compiler \
    u-boot-tools \
    && rm -rf /var/lib/apt/lists/*

# Install PYNQ (if available)
USER hdcuser
RUN pip install pynq || echo "PYNQ installation failed - FPGA tools may not be available"

# Create workspace for FPGA development
RUN mkdir -p /home/hdcuser/fpga-workspace
WORKDIR /home/hdcuser/fpga-workspace

# Default command
CMD ["bash"]

# =============================================================================
# Documentation stage
# =============================================================================
FROM development as docs

# Install documentation dependencies
RUN pip install \
    sphinx-autobuild \
    sphinx-book-theme \
    sphinxcontrib-bibtex \
    sphinx-copybutton

# Build documentation
WORKDIR /home/hdcuser/hd-compute-toolkit
RUN make docs || echo "Documentation build failed"

# Expose port for documentation server
EXPOSE 8000

# Serve documentation
CMD ["python", "-m", "http.server", "8000", "--directory", "docs/_build/html"]

# =============================================================================
# Benchmark stage
# =============================================================================
FROM production as benchmark

# Install benchmark dependencies
RUN pip install \
    matplotlib \
    seaborn \
    pandas \
    psutil \
    py-spy

# Create results directory
RUN mkdir -p /home/hdcuser/benchmark-results

# Default command runs benchmarks
CMD ["python", "-m", "hd_compute.benchmarks", "--output-dir", "/home/hdcuser/benchmark-results"]