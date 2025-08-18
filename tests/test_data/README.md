# Test Data Directory

This directory contains test datasets and reference data for HD-Compute-Toolkit testing.

## Directory Structure

```
test_data/
├── README.md                 # This file
├── reference/                # Reference implementations and expected outputs
│   ├── bundling_results.npz  # Expected results for bundling operations
│   ├── binding_results.npz   # Expected results for binding operations
│   └── similarity_matrix.npz # Expected similarity matrices
├── speech/                   # Speech command test data
│   ├── samples/              # Audio sample files
│   ├── mfcc_features.npz     # Pre-computed MFCC features
│   └── labels.txt            # Ground truth labels
├── benchmarks/               # Benchmark reference data
│   ├── performance_baselines.json
│   ├── memory_profiles.json
│   └── hardware_specs.json
└── synthetic/                # Synthetically generated test data
    ├── hypervectors/         # Pre-generated hypervector sets
    ├── sequences/            # Test sequences for temporal encoding
    └── hierarchies/          # Hierarchical concept structures
```

## Data Generation

Test data is generated using reproducible seeds to ensure consistent results across test runs:

- **Seed 42**: Primary seed for most test data
- **Seed 123**: Secondary seed for validation data
- **Seed 456**: Tertiary seed for stress testing

## File Formats

### NumPy Archives (.npz)
Binary data stored in compressed NumPy format for efficient loading:
- Hypervectors: `int8` arrays with values {0, 1}
- Features: `float32` arrays normalized to [-1, 1]
- Labels: `int32` arrays for classification

### JSON Files
Configuration and metadata in human-readable format:
- Performance baselines with timing and memory usage
- Hardware specifications for benchmark comparison
- Test parameters and expected outcomes

### Text Files
Simple text format for labels and metadata:
- One item per line
- UTF-8 encoding
- No special characters in identifiers

## Usage in Tests

```python
import numpy as np
from pathlib import Path

# Load reference data
test_data_dir = Path(__file__).parent / "test_data"
reference_data = np.load(test_data_dir / "reference" / "bundling_results.npz")

# Verify test results against reference
assert np.allclose(computed_result, reference_data['expected'])
```

## Maintenance

- Update reference data when algorithms change
- Version control large binary files using Git LFS
- Document data provenance and generation methods
- Regular validation against independent implementations