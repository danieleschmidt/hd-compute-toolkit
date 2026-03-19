# hd-compute-toolkit

A clean, dependency-light Python toolkit for **Hyperdimensional Computing (HDC)** —
the neuromorphic computing paradigm that encodes information into high-dimensional
binary or bipolar vectors and performs learning through associative memory.

> **No PyTorch. No JAX. No bloat.** Just NumPy and a principled implementation of
> the core HDC algebra.

---

## What is Hyperdimensional Computing?

HDC (also called Vector Symbolic Architecture, or VSA) is inspired by how the brain
represents and manipulates information. The key insight: in very high-dimensional
spaces (D ≈ 10 000), random vectors are nearly orthogonal to each other, and two
simple operations — **bundle** and **bind** — are enough to encode compositional
structure.

| Operation | Meaning | Binary | Bipolar |
|-----------|---------|--------|---------|
| **Bundle** | "similar to all of these" | majority vote | element-wise sum → sign |
| **Bind** | "association / role-filler" | XOR | element-wise multiply |

These two operations, combined with a **random-projection encoder** and an
**associative memory** (nearest-prototype lookup), yield a complete learning system
that is:

- **One-shot** — a single pass can build useful prototypes
- **Neuromorphic-friendly** — binary/bipolar vectors, no floating-point multiplication at inference
- **Noise-tolerant** — classification degrades gracefully under bit-flip noise
- **Interpretable** — similarity is just dot-product / Hamming distance

---

## Installation

```bash
pip install numpy          # only dependency
pip install -e .           # install this package
```

Or just drop the `hdc/` directory into your project.

---

## Quick Start

```python
import numpy as np
from hdc import HDEncoder, HDClassifier, HDTrainer

# 1. Set up encoder and classifier
encoder    = HDEncoder(dim=10_000, bipolar=True, seed=42)
classifier = HDClassifier(dim=10_000, metric="cosine")
trainer    = HDTrainer(encoder, classifier, n_iter=20, verbose=True)

# 2. Train on labelled feature matrix
X_train = np.random.randn(200, 20)
y_train = ["cat"] * 100 + ["dog"] * 100

trainer.fit(X_train, y_train)

# 3. Classify new samples
X_test   = np.random.randn(10, 20)
hvs_test = np.stack([encoder.encode_continuous(x) for x in X_test])
preds    = classifier.predict_batch(hvs_test)
print(preds)
```

---

## API Reference

### `HDEncoder(dim=10_000, bipolar=True, seed=None)`

Encodes raw data into hypervectors.

| Method | Input | Description |
|--------|-------|-------------|
| `encode_continuous(x)` | float array | Random-projection encoding for continuous features |
| `encode_level(value, n_levels)` | int | Level codebook: adjacent levels share ~50% bits |
| `encode_sequence(symbols)` | list of str | Positional encoding: bind each symbol with its position HV |
| `random_hv()` | — | Sample a fresh random hypervector |

### `BundleOperation`

```python
from hdc import BundleOperation
bundle = BundleOperation()
result = bundle.bundle([hv1, hv2, hv3], threshold=True)
# result is similar to each input
```

### `BindOperation`

```python
from hdc import BindOperation
bind = BindOperation()
bound   = bind.bind(hv_role, hv_filler)    # dissimilar to both
unbound = bind.unbind(bound, hv_role)      # == hv_filler  (invertible)
```

### `HDClassifier(dim=10_000, metric="cosine")`

Associative memory for classification.

| Method | Description |
|--------|-------------|
| `add_class(label, hv)` | Store prototype for a class |
| `predict(query_hv)` | Nearest-prototype classification |
| `predict_batch(queries)` | Vectorised batch prediction |
| `predict_proba(query_hv)` | Softmax-normalised similarity scores |
| `similarity(query_hv)` | Raw similarity scores to all prototypes |

### `HDTrainer(encoder, classifier, n_iter=10, verbose=False)`

```python
trainer.fit(X_train, y_train)
```

Training algorithm:
1. Encode all samples → HVs
2. Bundle each class's HVs → initial prototype
3. Iteratively correct misclassified samples (add to true class, subtract from wrong class)
4. Binarise and store final prototypes

---

## Demo

```bash
python examples/classify_demo.py
```

Output (3-class synthetic dataset, D=10 000):

```
── HDC Training ──
[HDTrainer] Iter 1/20: train_acc=1.0000, errors=0
[HDTrainer] Perfect training accuracy — stopping early.

  HDC accuracy : 1.0000  (train 0.07s, infer 0.022s)
  k-NN accuracy: 1.0000  (infer 0.009s)

── Bundle & Bind sanity ──
  Bundled similarity to component a : +0.5068
  Bundled similarity to random HV   : -0.0184  (expected ≈ 0)
  Bind(a,b) similarity to a         : +0.0134  (expected ≈ 0)
  Unbind(Bind(a,b), b) recovery of a: +1.0000  (expected ≈ 1)

── Sequence encoding ──
  seq1 ↔ seq3 (same):    +1.0000  (expected ≈ 1)
  seq1 ↔ seq2 (reorder): +0.4944  (expected lower)
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
# 35 passed in 0.12s
```

---

## Background & References

- Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.* Cognitive Computation.
- Plate, T. (1995). *Holographic Reduced Representations.* IEEE Trans. Neural Networks.
- Rahimi, A. & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines.* NeurIPS.
- Imani, M. et al. (2019). *HDNN: A Cognitive Neural Network using Hyperdimensional Computing.* DATE.

---

## License

MIT © Daniel Schmidt
