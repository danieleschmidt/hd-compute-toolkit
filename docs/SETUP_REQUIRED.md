# Manual Setup Required

This document outlines the manual setup steps required to complete the HD-Compute-Toolkit SDLC implementation. These steps couldn't be automated during the initial setup due to GitHub App permission limitations.

## Overview

The automated SDLC implementation has completed 8 checkpoints covering:
- ✅ Project foundation and documentation
- ✅ Development environment and tooling  
- ✅ Testing infrastructure
- ✅ Build and containerization
- ✅ Monitoring and observability setup
- ✅ Workflow documentation and templates
- ✅ Metrics and automation setup
- ✅ Integration and final configuration

However, some tasks require **manual intervention by repository maintainers** due to GitHub permissions.

## Required Manual Actions

### 1. GitHub Workflows Setup

**Status**: 🔴 **REQUIRED**  
**Estimated Time**: 15 minutes  
**Prerequisites**: Admin access to repository

The following GitHub Action workflows need to be manually created:

#### Create `.github/workflows/ci.yml`
```bash
# Copy the template to the correct location
cp docs/workflows/examples/ci-complete.yml .github/workflows/ci.yml
```

#### Create additional workflow files
```bash
# Copy all workflow templates
cp docs/workflows/templates/*.yml .github/workflows/
```

#### Required workflow files:
- `.github/workflows/ci.yml` - Complete CI pipeline
- `.github/workflows/security.yml` - Security scanning  
- `.github/workflows/release.yml` - Automated releases
- `.github/workflows/benchmark.yml` - Performance benchmarks

### 2. Branch Protection Rules

**Status**: 🔴 **REQUIRED**  
**Estimated Time**: 10 minutes  
**Prerequisites**: Admin access to repository

Navigate to **Settings → Branches** and create protection rules for `main` branch:

#### Required Protection Rules:
- ✅ Require a pull request before merging
- ✅ Require approvals (minimum 1)
- ✅ Dismiss stale PR approvals when new commits are pushed
- ✅ Require review from code owners
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require linear history
- ✅ Include administrators

#### Required Status Checks:
- `quality` (Code quality & security)
- `test-unit` (Unit tests)  
- `test-integration` (Integration tests)
- `docs` (Documentation build)
- `docker` (Docker build & test)

### 3. Repository Settings

**Status**: 🔴 **REQUIRED**  
**Estimated Time**: 5 minutes  
**Prerequisites**: Admin access to repository

#### General Settings
Navigate to **Settings → General**:
- ✅ Allow squash merging
- ✅ Allow auto-merge
- ✅ Automatically delete head branches
- ❌ Allow merge commits (disable)
- ❌ Allow rebase merging (disable)

#### Security Settings
Navigate to **Settings → Security & analysis**:
- ✅ Enable Dependabot alerts
- ✅ Enable Dependabot security updates
- ✅ Enable Dependabot version updates
- ✅ Enable Secret scanning
- ✅ Enable Push protection

### 4. GitHub App Permissions

**Status**: 🟡 **OPTIONAL**  
**Estimated Time**: 20 minutes  
**Prerequisites**: Organization admin access

For full automation, the GitHub App needs additional permissions:

#### Required Permissions:
- **Repository permissions**:
  - Actions: Write
  - Administration: Write
  - Checks: Write
  - Contents: Write
  - Issues: Write
  - Metadata: Read
  - Pages: Write
  - Pull requests: Write
  - Security events: Write
  - Statuses: Write

#### To Update Permissions:
1. Go to GitHub App settings
2. Navigate to Permissions & webhooks
3. Update repository permissions as listed above
4. Save changes and reinstall app on repository

### 5. Secrets and Environment Variables

**Status**: 🔴 **REQUIRED**  
**Estimated Time**: 10 minutes  
**Prerequisites**: Admin access to repository

Navigate to **Settings → Secrets and variables → Actions**:

#### Required Secrets:
```bash
# CodeCov integration
CODECOV_TOKEN=<your_codecov_token>

# PyPI publishing (for releases)
PYPI_API_TOKEN=<your_pypi_token>
TEST_PYPI_API_TOKEN=<your_test_pypi_token>

# Docker Hub (optional)
DOCKER_USERNAME=<your_docker_username>
DOCKER_PASSWORD=<your_docker_password>

# Security scanning (optional)
SNYK_TOKEN=<your_snyk_token>
```

#### Required Variables:
```bash
# Python version matrix
PYTHON_VERSIONS='["3.8", "3.9", "3.10", "3.11"]'

# CUDA version
CUDA_VERSION=12.1

# Performance thresholds
MAX_BUILD_TIME=300
MIN_COVERAGE=80
```

### 6. Self-Hosted Runners (GPU Testing)

**Status**: 🟡 **OPTIONAL**  
**Estimated Time**: 60 minutes  
**Prerequisites**: GPU hardware, Docker, NVIDIA drivers

For GPU testing, set up self-hosted runners:

#### Setup Commands:
```bash
# Install GitHub Actions runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure runner
./config.sh --url https://github.com/danieleschmidt/hd-compute-toolkit --token <YOUR_TOKEN>

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

#### Required Labels:
- `self-hosted`
- `gpu`
- `cuda`

### 7. Dependabot Configuration

**Status**: 🟡 **OPTIONAL**  
**Estimated Time**: 5 minutes  
**Prerequisites**: Admin access to repository

Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    assignees:
      - "danieleschmidt"
    reviewers:
      - "danieleschmidt"
    
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      
  - package-ecosystem: "github-actions"
    directory: "/.github/workflows"
    schedule:
      interval: "weekly"
```

### 8. Code Owners

**Status**: 🟡 **OPTIONAL**  
**Estimated Time**: 3 minutes  
**Prerequisites**: Repository access

Create `.github/CODEOWNERS`:
```bash
# Global owners
* @danieleschmidt

# Core HDC code
/hd_compute/core/ @danieleschmidt
/hd_compute/torch/ @danieleschmidt  
/hd_compute/jax/ @danieleschmidt

# Hardware acceleration
/hd_compute/kernels/ @danieleschmidt
/hd_compute/fpga/ @danieleschmidt
/hd_compute/vulkan/ @danieleschmidt

# Documentation
/docs/ @danieleschmidt
*.md @danieleschmidt

# CI/CD
/.github/ @danieleschmidt
/docker* @danieleschmidt
/Makefile @danieleschmidt

# Configuration
pyproject.toml @danieleschmidt
.pre-commit-config.yaml @danieleschmidt
```

### 9. External Service Integrations

**Status**: 🟡 **OPTIONAL**  
**Estimated Time**: 30 minutes  
**Prerequisites**: Service accounts

#### CodeCov Setup:
1. Visit https://codecov.io/
2. Connect GitHub account
3. Add repository
4. Copy token to GitHub secrets

#### ReadTheDocs Setup:
1. Visit https://readthedocs.org/
2. Import repository
3. Configure build settings
4. Enable webhook integration

## Verification Steps

After completing the manual setup:

### 1. Test CI Pipeline
```bash
# Create a test branch and PR
git checkout -b test-ci-setup
echo "# Test" > test-file.md
git add test-file.md
git commit -m "test: verify CI pipeline"
git push -u origin test-ci-setup
```

### 2. Verify Branch Protection
- Try to push directly to `main` (should be blocked)
- Verify status checks are required for PR merge

### 3. Check Automation
- Verify Dependabot creates PRs for dependency updates
- Test pre-commit hooks on local development
- Confirm Docker builds work correctly

### 4. Validate Documentation
- Check that documentation builds successfully
- Verify API docs are generated correctly
- Test example code in README

## Troubleshooting

### Common Issues

#### CI Workflows Not Triggering
- Verify workflow files are in `.github/workflows/`
- Check YAML syntax with `yamllint`
- Ensure proper indentation and formatting

#### Tests Failing
- Check Python version compatibility
- Verify all dependencies are installed
- Review test configuration in `pyproject.toml`

#### Docker Build Issues
- Ensure `.dockerignore` is properly configured
- Check base image availability
- Verify CUDA compatibility for GPU tests

#### Permission Errors
- Confirm GitHub App has necessary permissions
- Check repository access levels for team members
- Verify branch protection rules are not overly restrictive

### Getting Help

If you encounter issues during manual setup:

1. **Check Documentation**: Review relevant docs in `/docs/` directory
2. **GitHub Issues**: Search existing issues or create new one
3. **Community Support**: Join discussions in repository
4. **Professional Support**: Contact maintainers for enterprise needs

## Success Validation

✅ **Setup Complete When:**
- All CI workflows pass successfully
- Branch protection prevents direct pushes to main
- Pre-commit hooks run on every commit
- Documentation builds and deploys correctly
- Docker images build successfully
- Security scanning runs without errors
- Dependabot creates update PRs
- Performance benchmarks execute properly

---

**Estimated Total Time**: 2-3 hours (depending on optional components)  
**Priority**: Complete required items first, then optional based on needs  
**Support**: Available through GitHub issues and community discussions