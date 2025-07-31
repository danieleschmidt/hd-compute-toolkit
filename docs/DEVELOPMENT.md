# Development Guide

This guide covers setting up a development environment and contributing to HD-Compute-Toolkit.

## Prerequisites

- Python 3.8+ 
- PyTorch 1.12+ or JAX 0.3+
- CUDA toolkit (for GPU acceleration)
- Git

## Development Setup

### 1. Clone and Setup Environment

```bash
git clone https://github.com/yourusername/hd-compute-toolkit.git
cd hd-compute-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests
pytest

# Check code formatting
black --check .
isort --check-only .

# Type checking
mypy hd_compute
```

## Project Structure

```
hd-compute-toolkit/
├── hd_compute/           # Main package
│   ├── core/            # Abstract base classes
│   ├── torch/           # PyTorch implementations
│   ├── jax/             # JAX implementations  
│   ├── kernels/         # Hardware acceleration
│   ├── memory/          # Memory structures
│   └── applications/    # Example applications
├── tests/               # Test suite
├── docs/                # Documentation
├── benchmarks/          # Performance benchmarks
└── examples/            # Usage examples
```

## Testing

Run the full test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=hd_compute --cov-report=html
```

## Performance Benchmarking

```bash
python -m benchmarks.run_benchmarks --dim 16000
```

## Building Documentation

```bash
cd docs/
make html
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release after merge
5. Build and publish to PyPI