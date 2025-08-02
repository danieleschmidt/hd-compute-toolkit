# Test Fixtures

This directory contains test fixtures and data files used by the HD-Compute-Toolkit test suite.

## Structure

```
fixtures/
├── datasets/           # Test datasets
│   ├── mock_speech_commands.npz
│   └── synthetic_data/
├── hypervectors/       # Pre-computed hypervector examples
├── benchmarks/         # Benchmark reference data
└── configs/           # Test configuration files
```

## Datasets

### Mock Speech Commands (`mock_speech_commands.npz`)
- Synthetic speech command data for testing
- Features: 100 samples x 50 timesteps x 13 MFCC features
- Labels: 10 classes (command_0 to command_9)
- Generated automatically by test fixtures

### Synthetic Data
Various synthetic datasets for testing different HDC operations:
- Random hypervector collections
- Sequence data for temporal encoding tests
- Classification datasets with known ground truth

## Hypervectors

Pre-computed hypervectors for consistency testing:
- Reference hypervectors with known properties
- Test vectors for specific similarity values
- Validation sets for regression testing

## Benchmarks

Reference data for performance testing:
- Expected operation timings
- Memory usage baselines
- Accuracy targets for applications

## Usage

Test fixtures are automatically loaded by pytest through `conftest.py`. They can be accessed in tests using fixture parameters:

```python
def test_example(mock_speech_data, test_data_dir):
    # mock_speech_data provides path to speech dataset
    # test_data_dir provides path to fixtures directory
    pass
```

## Adding New Fixtures

1. Create the fixture file in the appropriate subdirectory
2. Add loading logic to `conftest.py` if needed
3. Document the fixture format and usage
4. Update this README

## Data Generation

Some fixtures are generated dynamically during test runs to ensure reproducibility and avoid storing large binary files in the repository. See `conftest.py` for generation logic.