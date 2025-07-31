# Contributing to HD-Compute-Toolkit

Thank you for your interest in contributing to HD-Compute-Toolkit! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/yourusername/hd-compute-toolkit.git
   cd hd-compute-toolkit
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass:
   ```bash
   pytest
   ```

3. Format your code:
   ```bash
   black .
   isort .
   ```

4. Run type checking:
   ```bash
   mypy hd_compute
   ```

5. Commit your changes and push to your fork:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   git push origin feature/your-feature-name
   ```

6. Create a pull request from your fork to the main repository.

## Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 88)
- Use isort for import sorting
- Include type hints for all functions
- Write docstrings for all public functions and classes

## Testing

- Write unit tests for all new functionality
- Ensure test coverage remains above 90%
- Test both PyTorch and JAX implementations
- Include performance benchmarks for core operations

## Pull Request Guidelines

- Provide a clear description of the changes
- Include relevant issue numbers
- Ensure all tests pass
- Update documentation if needed
- Add yourself to the contributors list

## Reporting Issues

When reporting bugs, please include:
- Python version and OS
- PyTorch/JAX versions
- Hardware specifications (GPU/FPGA if relevant)
- Minimal code to reproduce the issue
- Expected vs actual behavior

## Questions?

Feel free to open an issue for questions or join our discussions!