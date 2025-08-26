# Multi-stage Docker build for HD-Compute-Toolkit
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip wheel setuptools
RUN pip install build

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r hdcuser && useradd -r -g hdcuser hdcuser

# Set up application directory
WORKDIR /app
COPY hd_compute/ ./hd_compute/
COPY pyproject.toml README.md LICENSE ./

# Install application
RUN pip install -e .

# Set up data and logs directories
RUN mkdir -p /app/data /app/logs && \
    chown -R hdcuser:hdcuser /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hd_compute; print('Health check passed')" || exit 1

# Security: Run as non-root user
USER hdcuser

# Environment variables
ENV PYTHONPATH=/app
ENV HDC_DATA_DIR=/app/data
ENV HDC_LOGS_DIR=/app/logs
ENV HDC_ENV=production

# Default command
CMD ["python", "-m", "hd_compute.api.server"]

# Labels for metadata
LABEL maintainer="daniel@example.com"
LABEL version="1.0.0"
LABEL description="HD-Compute-Toolkit Production Image"
LABEL org.opencontainers.image.source="https://github.com/yourusername/hd-compute-toolkit"
