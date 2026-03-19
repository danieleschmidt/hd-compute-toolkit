"""Tests for BundleOperation and BindOperation."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdc.ops import BundleOperation, BindOperation


D = 1000  # smaller dim for fast tests


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def bipolar(rng, d=D):
    return rng.choice([-1, 1], size=d).astype(np.int8)


def binary(rng, d=D):
    return rng.integers(0, 2, size=d, dtype=np.int8)


# ── Bundle ────────────────────────────────────────────────────────────────────

class TestBundleOperation:
    def test_bundle_bipolar_shape(self, rng):
        bundle = BundleOperation()
        hvs = [bipolar(rng) for _ in range(5)]
        result = bundle.bundle(hvs)
        assert result.shape == (D,)

    def test_bundle_similar_to_components(self, rng):
        bundle = BundleOperation()
        hvs = [bipolar(rng) for _ in range(9)]
        result = bundle.bundle(hvs, threshold=True).astype(float)
        for hv in hvs:
            sim = np.dot(result, hv.astype(float)) / D
            # Should be noticeably positive (not orthogonal)
            assert sim > 0.1, f"Expected positive similarity, got {sim:.4f}"

    def test_bundle_random_hv_near_zero_similarity(self, rng):
        bundle = BundleOperation()
        hvs = [bipolar(rng) for _ in range(9)]
        result = bundle.bundle(hvs, threshold=True).astype(float)
        rnd = bipolar(rng).astype(float)
        sim = abs(np.dot(result, rnd)) / D
        assert sim < 0.15, f"Expected near-zero similarity with random HV, got {sim:.4f}"

    def test_bundle_binary(self, rng):
        bundle = BundleOperation()
        hvs = [binary(rng) for _ in range(5)]
        result = bundle.bundle(hvs, threshold=True)
        assert set(np.unique(result)).issubset({0, 1})

    def test_bundle_empty_raises(self):
        bundle = BundleOperation()
        with pytest.raises(ValueError):
            bundle.bundle([])


# ── Bind ──────────────────────────────────────────────────────────────────────

class TestBindOperation:
    def test_bind_bipolar_invertible(self, rng):
        bind = BindOperation()
        a = bipolar(rng)
        b = bipolar(rng)
        bound   = bind.bind(a, b)
        unbound = bind.unbind(bound, b)
        sim = np.dot(unbound.astype(float), a.astype(float)) / D
        assert abs(sim - 1.0) < 1e-6, f"Unbind should recover a exactly; got sim={sim}"

    def test_bind_bipolar_dissimilar_to_inputs(self, rng):
        bind = BindOperation()
        a = bipolar(rng)
        b = bipolar(rng)
        bound = bind.bind(a, b).astype(float)
        sim_a = abs(np.dot(bound, a.astype(float))) / D
        sim_b = abs(np.dot(bound, b.astype(float))) / D
        assert sim_a < 0.1, f"Bound HV should be dissimilar to a; got {sim_a:.4f}"
        assert sim_b < 0.1, f"Bound HV should be dissimilar to b; got {sim_b:.4f}"

    def test_bind_binary_xor(self, rng):
        bind = BindOperation()
        a = binary(rng)
        b = binary(rng)
        bound = bind.bind(a, b)
        expected = np.bitwise_xor(a, b)
        np.testing.assert_array_equal(bound, expected)

    def test_bind_binary_invertible(self, rng):
        bind = BindOperation()
        a = binary(rng)
        b = binary(rng)
        bound   = bind.bind(a, b)
        unbound = bind.unbind(bound, b)
        np.testing.assert_array_equal(unbound, a)

    def test_bind_shape_mismatch_raises(self, rng):
        bind = BindOperation()
        a = bipolar(rng, d=100)
        b = bipolar(rng, d=200)
        with pytest.raises(ValueError):
            bind.bind(a, b)
