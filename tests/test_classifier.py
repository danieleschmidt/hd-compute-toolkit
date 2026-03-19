"""Tests for HDClassifier and HDTrainer."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdc.encoder import HDEncoder
from hdc.memory import HDClassifier
from hdc.trainer import HDTrainer


D = 2000
RNG = np.random.default_rng(42)


def make_enc():
    return HDEncoder(dim=D, bipolar=True, seed=42)


# ── HDClassifier ──────────────────────────────────────────────────────────────

class TestHDClassifier:
    def test_add_and_predict(self):
        enc = make_enc()
        clf = HDClassifier(dim=D, metric="cosine")
        hv_a = enc.random_hv()
        hv_b = enc.random_hv()
        clf.add_class("cat", hv_a)
        clf.add_class("dog", hv_b)
        # Predict exact prototype should return itself
        assert clf.predict(hv_a) == "cat"
        assert clf.predict(hv_b) == "dog"

    def test_predict_noisy_version(self):
        enc = make_enc()
        clf = HDClassifier(dim=D, metric="cosine")
        hv_a = enc.random_hv().astype(float)
        # Add slight noise: flip 1% of bits
        noisy = hv_a.copy()
        flip_idx = RNG.choice(D, size=D // 100, replace=False)
        noisy[flip_idx] *= -1
        clf.add_class("A", hv_a.astype(np.int8))
        clf.add_class("B", enc.random_hv())
        assert clf.predict(noisy.astype(np.int8)) == "A"

    def test_empty_raises(self):
        clf = HDClassifier(dim=D)
        hv = make_enc().random_hv()
        with pytest.raises(RuntimeError):
            clf.predict(hv)

    def test_classes_list(self):
        enc = make_enc()
        clf = HDClassifier(dim=D)
        clf.add_class("z", enc.random_hv())
        clf.add_class("a", enc.random_hv())
        assert clf.classes() == ["a", "z"]

    def test_predict_batch(self):
        enc = make_enc()
        clf = HDClassifier(dim=D)
        hv_a = enc.random_hv()
        hv_b = enc.random_hv()
        clf.add_class("A", hv_a)
        clf.add_class("B", hv_b)
        queries = np.stack([hv_a, hv_b, hv_a])
        preds = clf.predict_batch(queries)
        assert preds == ["A", "B", "A"]

    def test_predict_proba_sums_to_one(self):
        enc = make_enc()
        clf = HDClassifier(dim=D)
        clf.add_class("X", enc.random_hv())
        clf.add_class("Y", enc.random_hv())
        clf.add_class("Z", enc.random_hv())
        proba = clf.predict_proba(enc.random_hv())
        total = sum(proba.values())
        assert abs(total - 1.0) < 1e-6

    def test_hamming_metric(self):
        enc = HDEncoder(dim=D, bipolar=False, seed=0)  # binary encoder
        clf = HDClassifier(dim=D, metric="hamming")
        hv_a = enc.random_hv()
        hv_b = enc.random_hv()
        clf.add_class("A", hv_a)
        clf.add_class("B", hv_b)
        assert clf.predict(hv_a) == "A"


# ── HDTrainer ─────────────────────────────────────────────────────────────────

class TestHDTrainer:
    def _make_separable_data(self, n=100, d=10, n_classes=2):
        """Well-separated Gaussian clusters."""
        rng = np.random.default_rng(0)
        means = rng.standard_normal((n_classes, d)) * 5
        X, y = [], []
        for c, m in enumerate(means):
            X.append(m + rng.standard_normal((n, d)) * 0.3)
            y.extend([f"C{c}"] * n)
        return np.vstack(X), np.array(y)

    def test_high_accuracy_on_separable(self):
        X, y = self._make_separable_data(n=100, d=10, n_classes=2)
        enc = HDEncoder(dim=D, bipolar=True, seed=0)
        clf = HDClassifier(dim=D, metric="cosine")
        trainer = HDTrainer(enc, clf, n_iter=10, verbose=False)
        trainer.fit(X, y.tolist())

        hvs = np.stack([enc.encode_continuous(X[i]) for i in range(len(y))])
        preds = clf.predict_batch(hvs)
        acc = np.mean(np.array(preds) == y)
        assert acc >= 0.90, f"Expected ≥90% on separable data; got {acc:.4f}"

    def test_trainer_zero_iter(self):
        """Single-pass (no refinement) should still work."""
        X, y = self._make_separable_data(n=50, d=8, n_classes=2)
        enc = HDEncoder(dim=D, bipolar=True, seed=1)
        clf = HDClassifier(dim=D)
        trainer = HDTrainer(enc, clf, n_iter=0)
        trainer.fit(X, y.tolist())
        assert len(clf.classes()) == 2
