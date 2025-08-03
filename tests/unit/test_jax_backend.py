"""Tests for JAX HDC backend implementation."""

import pytest
import numpy as np

pytest.importorskip("jax")
import jax.numpy as jnp
import jax.random as random

from tests.conftest import assert_similarity_in_range


class TestHDComputeJAX:
    """Test JAX HDC backend functionality."""
    
    def test_initialization(self, medium_dimension):
        """Test HDComputeJAX initialization."""
        from hd_compute import HDComputeJAX
        
        key = random.PRNGKey(42)
        hdc = HDComputeJAX(dim=medium_dimension, key=key)
        assert hdc.dim == medium_dimension
        assert hdc.key is not None
    
    def test_random_hv_generation(self, jax_backend, medium_dimension):
        """Test random hypervector generation."""
        hv = jax_backend.random_hv()
        
        assert isinstance(hv, jnp.ndarray)
        assert hv.shape == (medium_dimension,)
        assert hv.dtype == jnp.float32
        
        # Check sparsity is approximately correct
        sparsity = jnp.mean(hv).item()
        assert 0.4 <= sparsity <= 0.6, f"Sparsity {sparsity} not around 0.5"
    
    def test_random_hv_batch(self, jax_backend, medium_dimension):
        """Test batch random hypervector generation."""
        batch_size = 5
        hvs = jax_backend.random_hv(batch_size=batch_size)
        
        assert isinstance(hvs, jnp.ndarray)
        assert hvs.shape == (batch_size, medium_dimension)
        
        # Check that vectors are different
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                hamming_dist = jnp.sum(jnp.logical_xor(hvs[i].astype(bool), hvs[j].astype(bool))).item()
                assert hamming_dist > medium_dimension * 0.3, "Vectors too similar"
    
    def test_bundle_operation(self, jax_backend):
        """Test bundling operation."""
        hvs = jnp.stack([jax_backend.random_hv() for _ in range(5)])
        bundled = jax_backend.bundle(hvs)
        
        assert isinstance(bundled, jnp.ndarray)
        assert bundled.shape == hvs[0].shape
        assert bundled.dtype == jnp.float32
    
    def test_bind_operation(self, jax_backend):
        """Test binding operation."""
        hv1 = jax_backend.random_hv()
        hv2 = jax_backend.random_hv()
        
        bound = jax_backend.bind(hv1, hv2)
        
        assert isinstance(bound, jnp.ndarray)
        assert bound.shape == hv1.shape
        assert bound.dtype == jnp.float32
        
        # XOR property: bind(A, A) should be close to zero vector
        self_bound = jax_backend.bind(hv1, hv1)
        zero_ratio = jnp.mean(self_bound).item()
        assert zero_ratio < 0.1, "Self-binding should produce mostly zeros"
    
    def test_cosine_similarity(self, jax_backend):
        """Test cosine similarity computation."""
        hv1 = jax_backend.random_hv()
        hv2 = jax_backend.random_hv()
        
        # Self-similarity should be 1
        self_sim = jax_backend.cosine_similarity(hv1, hv1).item()
        assert abs(self_sim - 1.0) < 1e-6, f"Self-similarity {self_sim} != 1.0"
        
        # Different vectors should have similarity around 0
        diff_sim = jax_backend.cosine_similarity(hv1, hv2).item()
        assert_similarity_in_range(diff_sim)
        assert abs(diff_sim) < 0.2, f"Random vectors too similar: {diff_sim}"
    
    def test_hamming_distance(self, jax_backend, medium_dimension):
        """Test Hamming distance computation."""
        hv1 = jax_backend.random_hv()
        hv2 = jax_backend.random_hv()
        
        # Self-distance should be 0
        self_dist = jax_backend.hamming_distance(hv1, hv1).item()
        assert self_dist == 0, f"Self-distance {self_dist} != 0"
        
        # Different vectors should have distance around dim/2
        diff_dist = jax_backend.hamming_distance(hv1, hv2).item()
        expected_range = (medium_dimension * 0.3, medium_dimension * 0.7)
        assert expected_range[0] <= diff_dist <= expected_range[1], \
            f"Hamming distance {diff_dist} not in expected range {expected_range}"
    
    def test_permute_operation(self, jax_backend, medium_dimension):
        """Test permutation operation."""
        hv = jax_backend.random_hv()
        positions = 10
        
        permuted = jax_backend.permute(hv, positions)
        
        assert isinstance(permuted, jnp.ndarray)
        assert permuted.shape == hv.shape
        
        # Check that permutation actually moved elements
        assert not jnp.array_equal(hv, permuted), "Permutation didn't change vector"
        
        # Check that reverse permutation restores original
        restored = jax_backend.permute(permuted, -positions)
        assert jnp.array_equal(hv, restored), "Reverse permutation failed"
    
    def test_batch_cosine_similarity(self, jax_backend):
        """Test batch cosine similarity computation."""
        batch_size = 3
        hvs1 = jax_backend.random_hv(batch_size=batch_size)
        hvs2 = jax_backend.random_hv(batch_size=batch_size)
        
        similarities = jax_backend.batch_cosine_similarity(hvs1, hvs2)
        
        assert isinstance(similarities, jnp.ndarray)
        assert similarities.shape == (batch_size,)
        
        # All similarities should be in valid range
        for sim in similarities:
            assert_similarity_in_range(sim.item())
    
    def test_cleanup_operation(self, jax_backend):
        """Test cleanup operation with item memory."""
        # Create item memory
        clean_hvs = jax_backend.random_hv(batch_size=5)
        
        # Create noisy version of first vector
        noise = jax_backend.random_hv() * 0.1  # Small noise
        noisy_hv = clean_hvs[0] + noise
        
        cleaned = jax_backend.cleanup(noisy_hv, clean_hvs, k=1)
        
        assert isinstance(cleaned, jnp.ndarray)
        assert cleaned.shape == noisy_hv.shape
        
        # Cleaned vector should be one from the memory
        assert jnp.array_equal(cleaned, clean_hvs[0]), "Cleanup didn't return expected vector"
    
    def test_encode_sequence(self, jax_backend):
        """Test sequence encoding with positional information."""
        sequence = [jax_backend.random_hv() for _ in range(3)]
        
        encoded = jax_backend.encode_sequence(sequence)
        
        assert isinstance(encoded, jnp.ndarray)
        assert encoded.shape == sequence[0].shape
        
        # Encoded sequence should be different from individual elements
        for hv in sequence:
            similarity = jax_backend.cosine_similarity(encoded, hv).item()
            assert abs(similarity) < 0.5, "Encoded sequence too similar to individual elements"
    
    def test_create_item_memory(self, jax_backend):
        """Test item memory creation."""
        items = ['apple', 'banana', 'cherry']
        
        memory, item_to_index = jax_backend.create_item_memory(items)
        
        assert isinstance(memory, jnp.ndarray)
        assert memory.shape == (len(items), jax_backend.dim)
        assert isinstance(item_to_index, dict)
        assert set(item_to_index.keys()) == set(items)
        assert list(item_to_index.values()) == list(range(len(items)))
    
    def test_jit_compilation(self, jax_backend):
        """Test that JIT compilation works correctly."""
        hv1 = jax_backend.random_hv()
        hv2 = jax_backend.random_hv()
        
        # These operations should be JIT compiled
        bound = jax_backend.bind(hv1, hv2)
        similarity = jax_backend.cosine_similarity(hv1, hv2)
        hamming_dist = jax_backend.hamming_distance(hv1, hv2)
        
        assert isinstance(bound, jnp.ndarray)
        assert isinstance(similarity, jnp.ndarray)
        assert isinstance(hamming_dist, jnp.ndarray)
    
    def test_random_key_updates(self, medium_dimension):
        """Test that random key is properly updated."""
        from hd_compute import HDComputeJAX
        
        hdc = HDComputeJAX(dim=medium_dimension, key=random.PRNGKey(42))
        
        # Generate multiple vectors and check they're different
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        hv3 = hdc.random_hv()
        
        # All should be different
        assert not jnp.array_equal(hv1, hv2)
        assert not jnp.array_equal(hv2, hv3)
        assert not jnp.array_equal(hv1, hv3)
    
    @pytest.mark.parametrize("sparsity", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_sparsity_control(self, jax_backend, sparsity):
        """Test that sparsity parameter controls hypervector density."""
        hv = jax_backend.random_hv(sparsity=sparsity)
        actual_sparsity = jnp.mean(hv).item()
        
        # Allow some tolerance due to randomness
        assert abs(actual_sparsity - sparsity) < 0.1, \
            f"Actual sparsity {actual_sparsity} too far from target {sparsity}"
    
    def test_reproducibility(self, medium_dimension):
        """Test that same key produces same results."""
        from hd_compute import HDComputeJAX
        
        key = random.PRNGKey(42)
        hdc1 = HDComputeJAX(dim=medium_dimension, key=key)
        hdc2 = HDComputeJAX(dim=medium_dimension, key=key)
        
        hv1 = hdc1.random_hv()
        hv2 = hdc2.random_hv()
        
        # Should be the same
        assert jnp.array_equal(hv1, hv2), "Same key should produce same results"