#!/bin/bash
set -e

echo "Setting up HD-Compute-Toolkit development environment..."

# Install development dependencies
echo "Installing Python dependencies..."
pip install -e ".[dev]"

# Setup pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p models
mkdir -p notebooks

# Set up Jupyter Lab configuration
echo "Configuring Jupyter Lab..."
jupyter lab --generate-config --allow-root || true
echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py

# Install additional development tools
echo "Installing additional development tools..."
pip install jupyterlab-git jupyter-black ipywidgets

# Set up Git configuration (if not already configured)
if [ -z "$(git config --global user.name)" ]; then
    echo "Setting up Git configuration..."
    git config --global user.name "HD-Compute Developer"
    git config --global user.email "developer@hd-compute.dev"
fi

# Install CUDA development tools if available
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, installing additional GPU development tools..."
    pip install cupy-cuda11x
fi

# Create example notebooks directory
mkdir -p examples/notebooks

echo "Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  make test       - Run test suite"
echo "  make lint       - Run code quality checks"
echo "  make format     - Format code with black and isort"
echo "  make docs       - Build documentation"
echo "  jupyter lab     - Start Jupyter Lab server"
echo ""