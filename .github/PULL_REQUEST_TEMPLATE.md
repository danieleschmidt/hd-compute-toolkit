# Pull Request

## Summary

<!-- Provide a brief description of the changes in this PR -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring (no functional changes)
- [ ] Test improvements
- [ ] CI/CD improvements
- [ ] Other (please describe):

## Related Issues

<!-- Link any related issues using the format: Fixes #123, Closes #456, Relates to #789 -->

- Fixes #
- Closes #
- Relates to #

## Changes Made

<!-- Provide a detailed description of the changes -->

### Core Changes
- 
- 
- 

### API Changes
- 
- 
- 

### Documentation Changes
- 
- 
- 

## Breaking Changes

<!-- If this is a breaking change, describe what breaks and the migration path -->

- [ ] This PR introduces breaking changes
- [ ] Migration guide has been provided
- [ ] Deprecation warnings have been added

**Breaking changes description:**


**Migration path:**


## Testing

<!-- Describe the tests you ran to verify your changes -->

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] End-to-end tests added/updated
- [ ] Performance tests added/updated
- [ ] GPU tests added/updated (if applicable)
- [ ] FPGA tests added/updated (if applicable)

### Test Results
```bash
# Paste relevant test output here
```

### Performance Impact
<!-- If applicable, provide benchmark results -->

- [ ] Performance benchmarks run
- [ ] No significant performance regression
- [ ] Performance improvement documented

**Benchmark results:**
```
# Paste benchmark results here
```

## Hardware Compatibility

<!-- Mark all that apply -->

- [ ] CPU (tested)
- [ ] CUDA GPU (tested)
- [ ] FPGA (tested)
- [ ] Vulkan (tested)
- [ ] TPU (tested)
- [ ] Not applicable

## Checklist

### Code Quality
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

### Documentation
- [ ] I have made corresponding changes to the documentation
- [ ] I have updated the CHANGELOG.md file
- [ ] I have added docstrings to new functions/classes
- [ ] I have updated type hints where applicable

### Dependencies
- [ ] I have checked that my changes don't introduce unnecessary dependencies
- [ ] If new dependencies are added, they are justified and documented
- [ ] Dependencies are pinned to appropriate versions

### Security
- [ ] I have considered security implications of my changes
- [ ] No sensitive information (API keys, passwords, etc.) is exposed
- [ ] Input validation is appropriate for new functionality

### Performance
- [ ] I have considered the performance impact of my changes
- [ ] Memory usage has been considered for large-scale operations
- [ ] GPU memory usage has been optimized (if applicable)

## Screenshots/Examples

<!-- If applicable, add screenshots or code examples to help explain your changes -->

### Before
```python
# Example of old behavior/API
```

### After
```python
# Example of new behavior/API
```

## Deployment Notes

<!-- Any special deployment considerations -->

- [ ] No deployment changes required
- [ ] Database migrations required
- [ ] Configuration changes required
- [ ] Environment variable changes required

**Deployment instructions:**


## Reviewer Notes

<!-- Any specific areas you'd like reviewers to focus on -->

**Areas of focus:**
- 
- 
- 

**Questions for reviewers:**
- 
- 
- 

## Additional Context

<!-- Add any other context about the pull request here -->

---

## Review Checklist (for reviewers)

- [ ] Code follows project conventions
- [ ] Tests are comprehensive and pass
- [ ] Documentation is clear and complete
- [ ] Performance implications are acceptable
- [ ] Security considerations are addressed
- [ ] Breaking changes are justified and documented
- [ ] The change is ready for production