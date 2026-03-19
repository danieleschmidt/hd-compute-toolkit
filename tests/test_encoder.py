"""Tests for HDEncoder."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdc.encoder import HDEncoder


D = 2000  # smaller dim for fast tests


@pytest.fixture
def enc():
    return HDEncoder(dim=D, bipolar=True, seed=7)


@pytest.fixture
def enc_bin():
    return HDEncoder(dim=D, bipolar=False, seed=7)


# ── random_hv ─────────────────────────────────────────────────────────────────

class TestRandomHV:
    def test_shape(self, enc):
        hv = enc.random_hv()
        assert hv.shape == (D,)

    def test_bipolar_values(self, enc):
        hv = enc.random_hv()
        assert set(np.unique(hv)).issubset({-1, 1})

    def test_binary_values(self, enc_bin):
        hv = enc_bin.random_hv()
        assert set(np.unique(hv)).issubset({0, 1})

    def test_near_orthogonal_pair(self, enc):
        hv1 = enc.random_hv().astype(float)
        hv2 = enc.random_hv().astype(float)
        sim = abs(np.dot(hv1, hv2)) / D
        assert sim < 0.1


# ── encode_continuous ─────────────────────────────────────────────────────────

class TestEncodeContinuous:
    def test_shape(self, enc):
        x = np.random.randn(10)
        hv = enc.encode_continuous(x)
        assert hv.shape == (D,)

    def test_deterministic_same_input(self, enc):
        x = np.random.randn(5)
        hv1 = enc.encode_continuous(x)
        hv2 = enc.encode_continuous(x)
        np.testing.assert_array_equal(hv1, hv2)

    def test_similar_inputs_similar_hvs(self, enc):
        x = np.random.randn(10)
        x_noisy = x + np.random.randn(10) * 0.01
        hv1 = enc.encode_continuous(x).astype(float)
        hv2 = enc.encode_continuous(x_noisy).astype(float)
        sim = np.dot(hv1, hv2) / D
        # very similar inputs should have high cosine similarity
        assert sim > 0.8, f"Expected high similarity for near-identical inputs; got {sim:.4f}"

    def test_different_inputs_less_similar(self, enc):
        x1 = np.random.randn(10)
        x2 = np.random.randn(10) * 5  # scale up to make very different
        hv1 = enc.encode_continuous(x1).astype(float)
        hv2 = enc.encode_continuous(x2).astype(float)
        sim = abs(np.dot(hv1, hv2)) / D
        assert sim < 0.7


# ── encode_level ──────────────────────────────────────────────────────────────

class TestEncodeLevel:
    def test_shape(self, enc):
        hv = enc.encode_level(3, n_levels=10)
        assert hv.shape == (D,)

    def test_adjacent_levels_more_similar_than_distant(self, enc):
        n = 20
        hvs = [enc.encode_level(i, n_levels=n).astype(float) for i in range(n)]
        sim_adjacent = np.dot(hvs[0], hvs[1]) / D
        sim_distant  = np.dot(hvs[0], hvs[n - 1]) / D
        assert sim_adjacent > sim_distant

    def test_same_level_identical(self, enc):
        hv1 = enc.encode_level(5, n_levels=10)
        hv2 = enc.encode_level(5, n_levels=10)
        np.testing.assert_array_equal(hv1, hv2)

    def test_clips_out_of_range(self, enc):
        hv_neg = enc.encode_level(-5, n_levels=10)
        hv_zero = enc.encode_level(0, n_levels=10)
        np.testing.assert_array_equal(hv_neg, hv_zero)


# ── encode_sequence ────────────────────────────────────────────────────────────

class TestEncodeSequence:
    def test_shape(self, enc):
        hv = enc.encode_sequence(["a", "b", "c"])
        assert hv.shape == (D,)

    def test_same_sequence_identical(self, enc):
        hv1 = enc.encode_sequence(["x", "y", "z"])
        hv2 = enc.encode_sequence(["x", "y", "z"])
        np.testing.assert_array_equal(hv1, hv2)

    def test_different_order_different_hv(self, enc):
        hv1 = enc.encode_sequence(["a", "b", "c"]).astype(float)
        hv2 = enc.encode_sequence(["c", "b", "a"]).astype(float)
        sim = np.dot(hv1, hv2) / D
        # reordering should produce dissimilar HV
        assert sim < 0.9, f"Reordered sequence should differ; sim={sim:.4f}"

    def test_empty_sequence(self, enc):
        hv = enc.encode_sequence([])
        assert hv.shape == (D,)
        np.testing.assert_array_equal(hv, np.zeros(D, dtype=np.int8))
