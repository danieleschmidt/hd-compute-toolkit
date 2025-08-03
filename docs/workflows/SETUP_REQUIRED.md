# Manual Setup Required

Due to GitHub App permission limitations, the following workflow files must be manually created by repository maintainers.

## Required Actions

### 1. Copy Workflow Files

Copy all files from `docs/workflows/templates/` and `docs/workflows/examples/` to `.github/workflows/`:

```bash
mkdir -p .github/workflows
cp docs/workflows/templates/*.yml .github/workflows/
cp docs/workflows/examples/*.yml .github/workflows/
```

### 2. Configure Repository Secrets

Add the following secrets in GitHub repository settings (Settings → Secrets and variables → Actions):

#### Required Secrets
- `PYPI_API_TOKEN` - For publishing packages to PyPI
- `CODECOV_TOKEN` - For code coverage reporting
- `GITHUB_TOKEN` - Automatically provided by GitHub

#### Optional Secrets (for enhanced features)
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `DOCKER_HUB_USERNAME` - For Docker image publishing
- `DOCKER_HUB_TOKEN` - For Docker image publishing
- `AWS_ACCESS_KEY_ID` - For AWS deployment
- `AWS_SECRET_ACCESS_KEY` - For AWS deployment

### 3. Enable GitHub Actions

1. Go to repository Settings → Actions → General
2. Set "Actions permissions" to "Allow all actions and reusable workflows"
3. Set "Workflow permissions" to "Read and write permissions"
4. Check "Allow GitHub Actions to create and approve pull requests"

### 4. Configure Branch Protection Rules

Go to Settings → Branches and add protection rules for `main` branch:

#### Required Status Checks
- `CI / test (3.8)`
- `CI / test (3.9)`
- `CI / test (3.10)`
- `CI / test (3.11)`
- `CI / lint-and-type-check`
- `Security / codeql`
- `Security / dependency-scan`

#### Additional Rules
- ✅ Require a pull request before merging
- ✅ Require approvals (1)
- ✅ Dismiss stale PR approvals when new commits are pushed
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require conversation resolution before merging
- ✅ Include administrators

### 5. Setup Self-Hosted Runners (Optional)

For GPU and FPGA testing, configure self-hosted runners:

#### GPU Runner Setup
```bash
# On GPU-enabled machine
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.309.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.309.0/actions-runner-linux-x64-2.309.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.309.0.tar.gz

# Configure runner
./config.sh --url https://github.com/YOUR_ORG/hd-compute-toolkit --token YOUR_RUNNER_TOKEN --labels gpu,nvidia,cuda

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

#### Required Runner Labels
- `gpu` - For GPU-enabled machines
- `nvidia` - For NVIDIA GPU machines
- `multi-gpu` - For multi-GPU setups
- `fpga` - For FPGA-enabled machines
- `vulkan` - For Vulkan compute capable machines
- `benchmark` - For dedicated benchmark runners

### 6. Configure Renovate (Optional)

Create `.github/renovate.json` for automated dependency updates:

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": [
    "config:base",
    ":dependencyDashboard",
    ":semanticCommits",
    ":separatePatchReleases"
  ],
  "schedule": ["before 9am on monday"],
  "packageRules": [
    {
      "matchPackagePatterns": ["torch", "jax"],
      "groupName": "ML frameworks",
      "schedule": ["before 9am on the first day of the month"]
    },
    {
      "matchManagers": ["github-actions"],
      "groupName": "GitHub Actions",
      "automerge": true,
      "automergeType": "pr"
    }
  ],
  "prConcurrentLimit": 3,
  "prHourlyLimit": 1
}
```

### 7. Configure Issue Templates

Create `.github/ISSUE_TEMPLATE/` directory with templates:

#### Bug Report (`bug_report.yml`)
```yaml
name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "needs-triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  
  - type: input
    id: version
    attributes:
      label: Version
      description: What version of HD-Compute-Toolkit are you running?
      placeholder: ex. 0.1.0
    validations:
      required: true

  - type: dropdown
    id: device
    attributes:
      label: Device
      description: What device are you using?
      options:
        - CPU
        - CUDA
        - Vulkan
        - FPGA
    validations:
      required: true

  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true

  - type: textarea
    id: reproduce
    attributes:
      label: Steps to Reproduce
      description: How can we reproduce this issue?
      placeholder: |
        1. Import hd_compute
        2. Run operation...
        3. See error
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: |
        Please provide details about your environment.
        Tip: You can get this info by running `python -c "import hd_compute; hd_compute.print_env_info()"`
      placeholder: |
        - OS: Ubuntu 20.04
        - Python: 3.11.0
        - PyTorch: 2.1.0
        - CUDA: 12.1
    validations:
      required: true
```

### 8. Configure Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description

Brief description of changes made.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

- [ ] Unit tests pass (`make test`)
- [ ] Integration tests pass (`make test-integration`)
- [ ] Performance tests pass (`make test-performance`)
- [ ] Code quality checks pass (`make quality-gate`)

## GPU Testing (if applicable)

- [ ] CUDA tests pass
- [ ] Multi-GPU tests pass (if applicable)
- [ ] Performance benchmarks show no regression

## Documentation

- [ ] Code is self-documenting with clear variable names and structure
- [ ] Docstrings added/updated for new functions
- [ ] README updated (if applicable)
- [ ] CHANGELOG updated (if applicable)

## Additional Notes

Any additional information about the changes or special considerations for reviewers.

---

**Generated with [Claude Code](https://claude.ai/code)**
```

### 9. Repository Topics and Description

Update repository settings:
- **Description**: "High-performance hyperdimensional computing library for PyTorch and JAX with FPGA and Vulkan acceleration"
- **Topics**: `hyperdimensional-computing`, `pytorch`, `jax`, `cuda`, `fpga`, `vulkan`, `machine-learning`, `cognitive-computing`, `neuromorphic`
- **Homepage**: Link to documentation site

### 10. Security Settings

Configure security settings:
- Enable Dependabot alerts
- Enable Dependabot security updates
- Enable Code scanning alerts
- Configure private vulnerability reporting

## Verification Checklist

After manual setup, verify the following:

- [ ] All workflow files are in `.github/workflows/`
- [ ] Repository secrets are configured
- [ ] Branch protection rules are active
- [ ] Actions are enabled and have proper permissions
- [ ] Self-hosted runners are connected (if applicable)
- [ ] Issue and PR templates are working
- [ ] Security features are enabled
- [ ] Repository metadata is complete

## Support

If you encounter issues during setup:

1. Check GitHub Actions documentation
2. Review workflow logs for specific errors
3. Consult the [troubleshooting guide](../TROUBLESHOOTING.md)
4. Open an issue with the `setup` label

## Performance Considerations

- **Self-hosted runners**: Recommended for GPU/FPGA testing
- **Matrix builds**: Consider reducing matrix size for faster CI
- **Caching**: Python dependencies and Docker layers are cached
- **Parallel jobs**: Workflows are optimized for parallel execution

## Security Considerations

- **Secrets**: Never commit secrets to the repository
- **Permissions**: Use minimum required permissions for tokens
- **Dependencies**: Automated security scanning is enabled
- **Code analysis**: CodeQL analysis runs on all PRs

This setup provides a comprehensive CI/CD pipeline with security scanning, performance testing, and automated dependency management.