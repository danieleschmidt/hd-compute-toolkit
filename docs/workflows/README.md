# GitHub Actions Workflows

This directory contains templates and documentation for GitHub Actions workflows. Since Claude Code cannot directly create or modify GitHub Actions workflows, these templates provide the complete configuration needed for manual setup.

## Required Workflows

### 1. Main CI Workflow (`ci.yml`)
- Runs on every push and pull request
- Tests across Python 3.8-3.11 and PyTorch/JAX versions
- Includes linting, type checking, and security scanning
- Generates coverage reports and SBOM

### 2. Security Workflow (`security.yml`)
- Runs CodeQL analysis
- Performs dependency vulnerability scanning
- Checks for secrets in commits
- Generates security alerts

### 3. Release Workflow (`release.yml`)
- Automated release creation from tags
- Builds and publishes to PyPI
- Generates changelog
- Creates GitHub release with artifacts

### 4. Performance Benchmarking (`benchmark.yml`)
- Runs performance tests on GPU/CPU
- Tracks performance regressions
- Generates benchmark reports
- Compares against baseline metrics

### 5. Documentation Deployment (`docs.yml`)
- Builds and deploys documentation
- Updates API references
- Validates documentation links

## Setup Instructions

1. Copy workflow files from `docs/workflows/templates/` to `.github/workflows/`
2. Configure repository secrets (PyPI tokens, etc.)
3. Adjust matrix configurations for your needs
4. Enable GitHub Actions in repository settings

## Workflow Dependencies

- **Secrets Required**: `PYPI_API_TOKEN`, `CODECOV_TOKEN`
- **Permissions**: Contents: read, Actions: read, Security-events: write
- **Branch Protection**: Require status checks from CI workflow

For detailed configuration of each workflow, see the individual template files.