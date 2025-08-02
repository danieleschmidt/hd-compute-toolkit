# Getting Started with HD-Compute-Toolkit

This guide will help you get up and running with HD-Compute-Toolkit for hyperdimensional computing research and applications.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for performance)
- 8GB+ RAM for large-scale hypervector operations

## Installation

### Basic Installation

```bash
pip install hd-compute-toolkit
```

### Development Installation

```bash
git clone https://github.com/yourusername/hd-compute-toolkit
cd hd-compute-toolkit
pip install -e ".[dev]"
```

### Hardware Acceleration (Optional)

For FPGA support:
```bash
pip install hd-compute-toolkit[fpga]
```

For Vulkan acceleration:
```bash
pip install hd-compute-toolkit[vulkan]
```

## Your First HDC Program

### Basic Operations

```python
import torch
from hd_compute import HDCompute

# Initialize HDC context
hdc = HDCompute(dim=10000, device='cuda' if torch.cuda.is_available() else 'cpu')

# Create random hypervectors
apple = hdc.random_hv()
fruit = hdc.random_hv()
red = hdc.random_hv()

# Bind apple with its attributes
apple_concept = hdc.bind(apple, hdc.bundle([fruit, red]))

# Check similarity
similarity = hdc.cosine_similarity(apple_concept, apple)
print(f"Apple-concept similarity: {similarity:.3f}")
```

### Working with Sequences

```python
# Encode a sequence using position vectors
positions = [hdc.random_hv() for _ in range(5)]
sequence = ["I", "love", "hyperdimensional", "computing", "research"]

# Create sequence representation
encoded_sequence = hdc.bundle([
    hdc.bind(hdc.encode_symbol(word), pos) 
    for word, pos in zip(sequence, positions)
])

# Query for specific positions
query_pos_2 = hdc.bind(encoded_sequence, positions[2])
decoded = hdc.decode_symbol(query_pos_2)
print(f"Word at position 2: {decoded}")  # Should be "hyperdimensional"
```

## Key Concepts

### Hypervectors

Hypervectors are high-dimensional (typically 10,000+ dimensions) binary vectors that form the foundation of HDC:

```python
# Create hypervectors
hv1 = hdc.random_hv()          # Random hypervector
hv2 = hdc.zeros()              # Zero vector
hv3 = hdc.ones()               # All-ones vector

print(f"Dimensionality: {hv1.shape}")  # (10000,)
print(f"Data type: {hv1.dtype}")       # torch.bool or jax bool
```

### Core Operations

#### Bundling (Addition/Superposition)
Combines multiple hypervectors while preserving similarity to components:

```python
# Bundle multiple concepts
fruits = hdc.bundle([apple, orange, banana])

# Check if original concepts are retrievable
apple_sim = hdc.cosine_similarity(fruits, apple)
orange_sim = hdc.cosine_similarity(fruits, orange)
print(f"Apple similarity in bundle: {apple_sim:.3f}")
```

#### Binding (Multiplication/Association)
Creates associations between hypervectors:

```python
# Bind noun with adjective
red_apple = hdc.bind(apple, red)
green_apple = hdc.bind(apple, green)

# Bound vectors are dissimilar to originals
similarity = hdc.cosine_similarity(red_apple, apple)
print(f"Red-apple to apple similarity: {similarity:.3f}")  # Near 0.5
```

#### Permutation
Reorders hypervector elements for sequence encoding:

```python
# Create sequence with permutation
word1_pos1 = word1  # First position
word2_pos2 = hdc.permute(word2, 1)  # Second position  
word3_pos3 = hdc.permute(word3, 2)  # Third position

sequence = hdc.bundle([word1_pos1, word2_pos2, word3_pos3])
```

### Memory Structures

#### Item Memory
Maps symbols to hypervectors:

```python
# Create item memory
memory = hdc.create_item_memory()

# Store symbol associations
memory.store("cat", hdc.random_hv())
memory.store("dog", hdc.random_hv())

# Retrieve hypervector for symbol
cat_hv = memory.get("cat")
```

#### Associative Memory
Retrieves similar hypervectors:

```python
# Create associative memory with training data
train_data = [
    (concept1, label1),
    (concept2, label2),
    # ... more training pairs
]

am = hdc.create_associative_memory(train_data)

# Query with new concept
query_concept = hdc.bundle([feature1, feature2])
prediction = am.query(query_concept)
```

## Advanced Usage

### Backend Selection

```python
# Use PyTorch backend (default)
from hd_compute.torch import HDComputeTorch
hdc_torch = HDComputeTorch(dim=16000, device='cuda')

# Use JAX backend for TPU acceleration
from hd_compute.jax import HDComputeJAX
import jax
hdc_jax = HDComputeJAX(dim=16000, key=jax.random.PRNGKey(42))
```

### Hardware Acceleration

```python
# FPGA acceleration
from hd_compute.kernels import FPGAAccelerator
fpga = FPGAAccelerator(bitstream="hdc_kernel.bit")
result = fpga.bundle_batch(hypervectors, batch_size=1000)

# Vulkan compute shaders
from hd_compute.kernels import VulkanAccelerator  
vulkan = VulkanAccelerator(device_id=0)
result = vulkan.similarity_matrix(query_hvs, database_hvs)
```

### Performance Optimization

```python
# Batch operations for better performance
batch_size = 1000
hypervectors = [hdc.random_hv() for _ in range(batch_size)]

# Efficient batch bundling
bundled = hdc.bundle_batch(hypervectors)

# Memory pooling for reduced allocations
with hdc.memory_pool():
    results = []
    for i in range(1000):
        hv = hdc.random_hv()  # Uses pooled memory
        results.append(hdc.bind(hv, reference))
```

## Common Patterns

### Classification

```python
def hdc_classifier(features, classes, train_data):
    """Simple HDC classifier example"""
    # Create class prototypes
    class_prototypes = {}
    for label in classes:
        # Bundle all training examples for this class
        examples = [features[i] for i, y in train_data if y == label]
        class_prototypes[label] = hdc.bundle(examples)
    
    def predict(query):
        similarities = {
            label: hdc.cosine_similarity(query, prototype)
            for label, prototype in class_prototypes.items()
        }
        return max(similarities.keys(), key=lambda k: similarities[k])
    
    return predict
```

### Sequence Learning

```python
def encode_sequence(sequence, hdc):
    """Encode temporal sequence with position binding"""
    encoded = []
    for i, item in enumerate(sequence):
        position = hdc.generate_position_vector(i)
        item_hv = hdc.encode_symbol(item)
        encoded.append(hdc.bind(item_hv, position))
    
    return hdc.bundle(encoded)
```

## Next Steps

- Explore the [examples/](../../examples/) directory for complete applications
- Read the [Architecture Overview](../ARCHITECTURE.md) for system design details  
- Check out [Performance Guide](performance.md) for optimization tips
- Join our community discussions for questions and contributions

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce hypervector dimensions or batch size
2. **Slow Performance**: Ensure GPU acceleration is enabled
3. **Import Errors**: Verify all dependencies are installed correctly

### Getting Help

- GitHub Issues: Report bugs and feature requests
- Documentation: Complete API reference available
- Community: Join discussions for questions and tips