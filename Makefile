# HD-Compute-Toolkit Development Makefile
# Provides convenient shortcuts for common development tasks

.PHONY: help install install-dev test test-all lint format type-check security benchmark clean docs build publish

# Default target
help: ## Show this help message
	@echo "HD-Compute-Toolkit Development Commands"
	@echo "======================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install package in development mode
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

install-all: ## Install all optional dependencies
	pip install -e ".[dev,fpga,vulkan,docs]"

# Testing targets
test: ## Run basic test suite
	pytest tests/ -v

test-all: ## Run all tests including integration and performance
	pytest tests/ -v --cov=hd_compute --cov-report=html --cov-report=term-missing

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-performance: ## Run performance benchmarks
	pytest tests/performance/ -m benchmark -v

test-gpu: ## Run GPU-specific tests (requires CUDA)
	pytest tests/ -m gpu -v

test-memory: ## Run memory efficiency tests
	pytest tests/performance/ -m memory -v

# Code quality targets
lint: ## Run all linting tools
	flake8 hd_compute tests
	black --check hd_compute tests
	isort --check-only hd_compute tests

format: ## Auto-format code
	black hd_compute tests
	isort hd_compute tests

type-check: ## Run type checking
	mypy hd_compute

security: ## Run security analysis
	bandit -r hd_compute/ -f json -o bandit-report.json
	safety check

# Quality gates (used in CI)
quality-gate: lint type-check security ## Run all quality checks

# Benchmarking
benchmark: ## Run performance benchmarks
	python -m hd_compute.benchmarks --dim 16000 --device cpu

benchmark-gpu: ## Run GPU benchmarks (requires CUDA)
	python -m hd_compute.benchmarks --dim 16000 --device cuda

benchmark-all: ## Run comprehensive benchmark suite
	python -m hd_compute.benchmarks --comprehensive

# Documentation
docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build
	cd docs && make clean

# Build and publish
build: ## Build package distributions
	python -m build

publish-test: ## Publish to test PyPI
	twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
	twine upload dist/*

# Cleanup
clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean docs-clean ## Clean everything including docs

# Development utilities
setup-dev: install-dev ## Set up complete development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

check: quality-gate test ## Run all checks (quality + tests)

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Container development
docker-build: ## Build development Docker image
	docker build -t hd-compute-toolkit:dev .

docker-run: ## Run development container
	docker run -it --rm --gpus all -v $(PWD):/workspace hd-compute-toolkit:dev

# Terragon autonomous operations
terragon-analyze: ## Analyze repository with Terragon value discovery
	@echo "Running Terragon value discovery analysis..."
	@echo "This would integrate with the autonomous SDLC system"

terragon-execute: ## Execute next highest-value task
	@echo "Executing next highest-value task from Terragon backlog..."
	@echo "This would run the autonomous task execution"

# Performance profiling
profile: ## Profile performance with py-spy
	py-spy record -o profile.svg -d 60 -- python -m hd_compute.benchmarks

profile-memory: ## Profile memory usage
	mprof run python -m hd_compute.benchmarks
	mprof plot

# Dependency management
deps-update: ## Update dependencies
	pip-compile --upgrade requirements-dev.in
	pip-compile --upgrade requirements.in

deps-sync: ## Sync dependencies
	pip-sync requirements-dev.txt

# Release management
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version  
	bump2version minor

version-major: ## Bump major version
	bump2version major

# Environment info
env-info: ## Show environment information
	@echo "Python version: $(shell python --version)"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "JAX version: $(shell python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"