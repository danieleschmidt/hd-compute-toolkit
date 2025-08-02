# HD-Compute-Toolkit Production Dockerfile
# Multi-stage build for optimized production image

# ==============================================================================
# Stage 1: Build Environment
# ==============================================================================
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md ./
COPY hd_compute/ ./hd_compute/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install build wheel && \
    pip install .

# ==============================================================================
# Stage 2: Development Environment (for dev/test)
# ==============================================================================
FROM builder as development

# Install development dependencies
RUN pip install -e ".[dev,docs]"

# Install additional development tools
RUN pip install \
    ipython \
    jupyterlab \
    notebook \
    ipywidgets

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini .pre-commit-config.yaml ./

# Set up development environment
USER appuser
EXPOSE 8888 6006 8000

# Default command for development
CMD ["bash"]

# ==============================================================================
# Stage 3: Production Environment
# ==============================================================================
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HDC_ENV=production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# Copy built package from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application files
COPY --chown=appuser:appuser hd_compute/ ./hd_compute/
COPY --chown=appuser:appuser README.md LICENSE ./

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hd_compute; print('HDC toolkit ready')" || exit 1

# Expose port for API (if applicable)
EXPOSE 8000

# Default command
CMD ["python", "-c", "import hd_compute; print('HD-Compute-Toolkit container ready')"]

# ==============================================================================
# Stage 4: GPU-Enabled Environment
# ==============================================================================
FROM nvidia/cuda:12.2-devel-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HDC_DEFAULT_DEVICE=cuda

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create app user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# Copy and install application
COPY pyproject.toml README.md ./
COPY hd_compute/ ./hd_compute/

# Install PyTorch with CUDA support and other dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -e ".[dev]"

# Switch to non-root user
USER appuser

# Verify CUDA setup
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Health check for GPU
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

EXPOSE 8000 8888

CMD ["python", "-c", "import hd_compute; import torch; print(f'HD-Compute-Toolkit GPU container ready. CUDA: {torch.cuda.is_available()}')"]