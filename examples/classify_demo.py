#!/usr/bin/env python3
"""
HDC Classification Demo
=======================
Classifies a synthetic 3-class dataset using hyperdimensional computing and
compares accuracy against a k-Nearest Neighbours baseline.

Run:
    ~/anaconda3/bin/python3 examples/classify_demo.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from hdc import HDEncoder, HDClassifier, HDTrainer, BundleOperation, BindOperation


# ── reproducibility ────────────────────────────────────────────────────────────
RNG_SEED = 42
np.random.seed(RNG_SEED)


# ── synthetic dataset ──────────────────────────────────────────────────────────
def make_dataset(n_per_class=200, n_features=20, n_classes=3, noise=0.5):
    """3-class Gaussian blobs."""
    means = np.random.randn(n_classes, n_features) * 3
    X, y = [], []
    labels = [f"class_{c}" for c in range(n_classes)]
    for c, mean in enumerate(means):
        samples = mean + np.random.randn(n_per_class, n_features) * noise
        X.append(samples)
        y.extend([labels[c]] * n_per_class)
    X = np.vstack(X)
    y = np.array(y)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def train_test_split(X, y, test_ratio=0.3):
    n = len(y)
    n_test = int(n * test_ratio)
    idx = np.random.permutation(n)
    return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]


# ── k-NN baseline ─────────────────────────────────────────────────────────────
def knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for xq in X_test:
        dists = np.linalg.norm(X_train - xq, axis=1)
        nn_idx = np.argpartition(dists, k)[:k]
        labels, counts = np.unique(y_train[nn_idx], return_counts=True)
        preds.append(labels[counts.argmax()])
    return np.array(preds)


def accuracy(pred, true):
    return np.mean(pred == true)


# ── demo ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Hyperdimensional Computing — Classification Demo")
    print("=" * 60)

    D = 10_000
    N_CLASSES = 3
    N_PER_CLASS = 300
    N_FEATURES = 20

    print(f"\nDataset: {N_CLASSES} classes × {N_PER_CLASS} samples, {N_FEATURES} features")
    print(f"HD dimension: D = {D:,}\n")

    X, y = make_dataset(n_per_class=N_PER_CLASS, n_features=N_FEATURES, n_classes=N_CLASSES)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.3)

    # ── 1. HD encoding + training ──────────────────────────────────────────────
    print("── HDC Training ──")
    encoder    = HDEncoder(dim=D, bipolar=True, seed=RNG_SEED)
    classifier = HDClassifier(dim=D, metric="cosine")
    trainer    = HDTrainer(encoder, classifier, n_iter=20, verbose=True)

    t0 = time.perf_counter()
    trainer.fit(X_train, y_train.tolist())
    hdc_train_sec = time.perf_counter() - t0

    # ── 2. HD inference ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    hvs_test  = np.stack([encoder.encode_continuous(X_test[i]) for i in range(len(y_test))])
    hdc_preds = classifier.predict_batch(hvs_test)
    hdc_infer_sec = time.perf_counter() - t0

    hdc_acc = accuracy(np.array(hdc_preds), y_test)

    # ── 3. k-NN baseline ──────────────────────────────────────────────────────
    print("\n── k-NN Baseline ──")
    t0 = time.perf_counter()
    knn_preds = knn_predict(X_train, y_train, X_test, k=3)
    knn_sec   = time.perf_counter() - t0
    knn_acc   = accuracy(knn_preds, y_test)

    # ── 4. Results ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  HDC accuracy : {hdc_acc:.4f}  "
          f"(train {hdc_train_sec:.2f}s, infer {hdc_infer_sec:.3f}s)")
    print(f"  k-NN accuracy: {knn_acc:.4f}  (infer {knn_sec:.3f}s)")
    print("=" * 60)

    # ── 5. Bundle / Bind sanity check ─────────────────────────────────────────
    print("\n── Bundle & Bind sanity ──")
    bundle_op = BundleOperation()
    bind_op   = BindOperation()

    hv_a = encoder.random_hv()
    hv_b = encoder.random_hv()
    hv_c = encoder.random_hv()

    bundled = bundle_op.bundle([hv_a, hv_b, hv_c], threshold=True)
    # bundled should be similar to a, b, c
    sim_a = float(np.dot(bundled.astype(float), hv_a.astype(float))) / D
    sim_rnd = float(np.dot(bundled.astype(float), encoder.random_hv().astype(float))) / D
    print(f"  Bundled similarity to component a : {sim_a:+.4f}")
    print(f"  Bundled similarity to random HV   : {sim_rnd:+.4f}  (expected ≈ 0)")

    bound    = bind_op.bind(hv_a, hv_b)
    unbound  = bind_op.unbind(bound, hv_b)
    sim_recover = float(np.dot(unbound.astype(float), hv_a.astype(float))) / D
    sim_bound_a = float(np.dot(bound.astype(float),  hv_a.astype(float))) / D
    print(f"  Bind(a,b) similarity to a         : {sim_bound_a:+.4f}  (expected ≈ 0)")
    print(f"  Unbind(Bind(a,b), b) recovery of a: {sim_recover:+.4f}  (expected ≈ 1)")

    # ── 6. Sequence encoding example ─────────────────────────────────────────
    print("\n── Sequence encoding ──")
    seq1 = ["the", "cat", "sat"]
    seq2 = ["sat", "cat", "the"]   # same words, different order
    seq3 = ["the", "cat", "sat"]   # identical to seq1

    hv_seq1 = encoder.encode_sequence(seq1)
    hv_seq2 = encoder.encode_sequence(seq2)
    hv_seq3 = encoder.encode_sequence(seq3)

    def cos_sim(a, b):
        af, bf = a.astype(float), b.astype(float)
        return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))

    print(f"  seq1 ↔ seq3 (same):    {cos_sim(hv_seq1, hv_seq3):+.4f}  (expected ≈ 1)")
    print(f"  seq1 ↔ seq2 (reorder): {cos_sim(hv_seq1, hv_seq2):+.4f}  (expected lower)")

    print("\nDemo complete ✓")


if __name__ == "__main__":
    main()
