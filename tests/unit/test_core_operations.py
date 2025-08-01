"""Unit tests for core HDC operations."""

import pytest
import numpy as np
import torch
from typing import Any, Dict, List
from unittest.mock import Mock, patch

# Import core modules (these will be implemented later)
try:
    from hd_compute.core.hdc import HDComputeBase
    from hd_compute.core.operations import (
        random_hypervector,
        bundle_hypervectors, 
        bind_hypervectors,
        hamming_distance,
        cosine_similarity
    )
except ImportError:
    # Create mock implementations for testing infrastructure
    HDComputeBase = Mock
    random_hypervector = Mock(return_value=np.ones(1000, dtype=np.int8))
    bundle_hypervectors = Mock(return_value=np.ones(1000, dtype=np.int8))
    bind_hypervectors = Mock(return_value=np.ones(1000, dtype=np.int8))
    hamming_distance = Mock(return_value=0.5)
    cosine_similarity = Mock(return_value=0.8)


class TestRandomHypervectorGeneration:
    """Test random hypervector generation functionality."""
    
    @pytest.mark.unit
    def test_random_hv_shape(self, dimension_under_test: int):
        """Test that random hypervectors have correct shape."""
        hv = random_hypervector(dimension_under_test)
        assert hv.shape == (dimension_under_test,)
    
    @pytest.mark.unit
    def test_random_hv_binary_values(self, dimension_under_test: int):
        """Test that binary hypervectors contain only 0s and 1s."""
        hv = random_hypervector(dimension_under_test, encoding="binary")
        assert np.all(np.isin(hv, [0, 1]))
    
    @pytest.mark.unit 
    def test_random_hv_bipolar_values(self, dimension_under_test: int):
        """Test that bipolar hypervectors contain only -1s and 1s."""
        hv = random_hypervector(dimension_under_test, encoding="bipolar")
        assert np.all(np.isin(hv, [-1, 1]))
    
    @pytest.mark.unit
    def test_random_hv_sparsity(self):
        """Test that sparse hypervectors respect sparsity constraints."""
        sparsity = 0.1
        hv = random_hypervector(1000, sparsity=sparsity)
        actual_sparsity = np.sum(hv != 0) / len(hv)
        assert abs(actual_sparsity - sparsity) < 0.05  # Allow 5% tolerance
    
    @pytest.mark.unit
    def test_random_hv_reproducibility(self, random_seed: int):
        """Test that random hypervectors are reproducible with fixed seed."""
        np.random.seed(random_seed)
        hv1 = random_hypervector(1000)
        
        np.random.seed(random_seed)
        hv2 = random_hypervector(1000)
        
        assert np.array_equal(hv1, hv2)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    def test_random_hv_batch(self, batch_size: int):
        """Test batch generation of random hypervectors."""
        hvs = random_hypervector(1000, batch_size=batch_size)
        assert hvs.shape == (batch_size, 1000)


class TestBundlingOperations:
    """Test hypervector bundling (superposition) operations."""
    
    @pytest.mark.unit
    def test_bundle_two_vectors(self, binary_hypervector_1d: np.ndarray):
        """Test bundling of two hypervectors."""
        hv1 = binary_hypervector_1d
        hv2 = random_hypervector(len(hv1))
        
        bundled = bundle_hypervectors([hv1, hv2])
        assert bundled.shape == hv1.shape
    
    @pytest.mark.unit
    def test_bundle_multiple_vectors(self):
        """Test bundling of multiple hypervectors."""
        hvs = [random_hypervector(1000) for _ in range(10)]
        bundled = bundle_hypervectors(hvs)
        assert bundled.shape == (1000,)
    
    @pytest.mark.unit
    def test_bundle_batch_processing(self, binary_hypervector_batch: np.ndarray):
        """Test batch bundling operations."""
        bundled = bundle_hypervectors(binary_hypervector_batch, axis=0)
        assert bundled.shape == (binary_hypervector_batch.shape[1],)
    
    @pytest.mark.unit
    def test_bundle_commutativity(self):
        """Test that bundling is commutative."""
        hv1 = random_hypervector(1000)
        hv2 = random_hypervector(1000)
        
        bundle1 = bundle_hypervectors([hv1, hv2])
        bundle2 = bundle_hypervectors([hv2, hv1])
        
        assert np.array_equal(bundle1, bundle2)
    
    @pytest.mark.unit
    def test_bundle_associativity(self):
        """Test that bundling is associative."""
        hv1 = random_hypervector(1000)
        hv2 = random_hypervector(1000)
        hv3 = random_hypervector(1000)
        
        # (A + B) + C
        bundle1 = bundle_hypervectors([bundle_hypervectors([hv1, hv2]), hv3])
        # A + (B + C)
        bundle2 = bundle_hypervectors([hv1, bundle_hypervectors([hv2, hv3])])
        
        assert np.array_equal(bundle1, bundle2)


class TestBindingOperations:
    """Test hypervector binding (association) operations."""
    
    @pytest.mark.unit
    def test_bind_two_vectors(self, binary_hypervector_1d: np.ndarray):
        """Test binding of two hypervectors."""
        hv1 = binary_hypervector_1d
        hv2 = random_hypervector(len(hv1))
        
        bound = bind_hypervectors(hv1, hv2)
        assert bound.shape == hv1.shape
    
    @pytest.mark.unit
    def test_bind_commutativity(self):
        """Test that binding is commutative."""
        hv1 = random_hypervector(1000)
        hv2 = random_hypervector(1000)
        
        bind1 = bind_hypervectors(hv1, hv2)
        bind2 = bind_hypervectors(hv2, hv1)
        
        assert np.array_equal(bind1, bind2)
    
    @pytest.mark.unit
    def test_bind_associativity(self):
        """Test that binding is associative."""
        hv1 = random_hypervector(1000)
        hv2 = random_hypervector(1000)
        hv3 = random_hypervector(1000)
        
        # (A * B) * C
        bind1 = bind_hypervectors(bind_hypervectors(hv1, hv2), hv3)
        # A * (B * C)
        bind2 = bind_hypervectors(hv1, bind_hypervectors(hv2, hv3))
        
        assert np.array_equal(bind1, bind2)
    
    @pytest.mark.unit
    def test_bind_self_inverse(self):
        """Test that binding a vector with itself multiple times returns identity."""
        hv = random_hypervector(1000, encoding="bipolar")
        
        # Bind with itself (should be close to identity for bipolar)
        bound = bind_hypervectors(hv, hv)
        
        # For bipolar vectors, binding with self should give all 1s
        expected_ones = np.sum(bound == 1) / len(bound)
        assert expected_ones > 0.95  # Allow some tolerance
    
    @pytest.mark.unit
    def test_bind_distributivity_over_bundle(self):
        """Test that binding distributes over bundling."""
        hv1 = random_hypervector(1000)
        hv2 = random_hypervector(1000)
        hv3 = random_hypervector(1000)
        
        # A * (B + C)
        bundled = bundle_hypervectors([hv2, hv3])
        left_side = bind_hypervectors(hv1, bundled)
        
        # (A * B) + (A * C)
        bound1 = bind_hypervectors(hv1, hv2)
        bound2 = bind_hypervectors(hv1, hv3)
        right_side = bundle_hypervectors([bound1, bound2])
        
        # Should be approximately equal (not exact due to discrete operations)
        correlation = cosine_similarity(left_side, right_side)
        assert correlation > 0.8


class TestSimilarityMetrics:
    """Test similarity metrics for hypervectors."""
    
    @pytest.mark.unit
    def test_hamming_distance_identical_vectors(self, binary_hypervector_1d: np.ndarray):
        """Test Hamming distance between identical vectors."""
        distance = hamming_distance(binary_hypervector_1d, binary_hypervector_1d)
        assert distance == 0.0
    
    @pytest.mark.unit
    def test_hamming_distance_opposite_vectors(self):
        """Test Hamming distance between opposite vectors."""
        hv1 = np.ones(1000, dtype=np.int8)
        hv2 = np.zeros(1000, dtype=np.int8)
        
        distance = hamming_distance(hv1, hv2)
        assert distance == 1.0
    
    @pytest.mark.unit
    def test_hamming_distance_symmetry(self):
        """Test that Hamming distance is symmetric."""
        hv1 = random_hypervector(1000)
        hv2 = random_hypervector(1000)
        
        dist1 = hamming_distance(hv1, hv2)
        dist2 = hamming_distance(hv2, hv1)
        
        assert abs(dist1 - dist2) < 1e-10
    
    @pytest.mark.unit
    def test_cosine_similarity_identical_vectors(self, bipolar_hypervector_1d: np.ndarray):
        """Test cosine similarity between identical vectors."""
        similarity = cosine_similarity(bipolar_hypervector_1d, bipolar_hypervector_1d)
        assert abs(similarity - 1.0) < 1e-10
    
    @pytest.mark.unit
    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity between orthogonal vectors."""
        # Create two orthogonal bipolar vectors
        hv1 = np.array([1, 1, -1, -1] * 250, dtype=np.int8)
        hv2 = np.array([1, -1, 1, -1] * 250, dtype=np.int8)
        
        similarity = cosine_similarity(hv1, hv2)
        assert abs(similarity) < 0.1  # Should be close to 0
    
    @pytest.mark.unit
    def test_cosine_similarity_symmetry(self):
        """Test that cosine similarity is symmetric."""
        hv1 = random_hypervector(1000, encoding="bipolar")
        hv2 = random_hypervector(1000, encoding="bipolar")
        
        sim1 = cosine_similarity(hv1, hv2)
        sim2 = cosine_similarity(hv2, hv1)
        
        assert abs(sim1 - sim2) < 1e-10
    
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [16, 32])
    def test_batch_similarity_computation(self, batch_size: int):
        """Test batch computation of similarity metrics."""
        hvs1 = np.random.choice([0, 1], size=(batch_size, 1000)).astype(np.int8)
        hvs2 = np.random.choice([0, 1], size=(batch_size, 1000)).astype(np.int8)
        
        # This would test batch similarity computation when implemented
        # distances = hamming_distance(hvs1, hvs2, batch=True)
        # assert distances.shape == (batch_size,)


class TestHDComputeBase:
    """Test the base HDCompute class functionality."""
    
    @pytest.mark.unit
    def test_hdc_base_initialization(self):
        """Test HDComputeBase initialization."""
        if HDComputeBase is not Mock:
            hdc = HDComputeBase(dim=1000)
            assert hdc.dim == 1000
    
    @pytest.mark.unit
    def test_hdc_base_abstract_methods(self):
        """Test that HDComputeBase defines required abstract methods."""
        if HDComputeBase is not Mock:
            # This test verifies that concrete implementations must implement abstract methods
            with pytest.raises(TypeError):
                HDComputeBase(dim=1000)  # Should fail if abstract methods not implemented


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.unit
    def test_empty_bundle(self):
        """Test bundling empty list of hypervectors."""
        with pytest.raises((ValueError, TypeError)):
            bundle_hypervectors([])
    
    @pytest.mark.unit
    def test_mismatched_dimensions_bundle(self):
        """Test bundling hypervectors with mismatched dimensions."""
        hv1 = random_hypervector(1000)
        hv2 = random_hypervector(2000)
        
        with pytest.raises((ValueError, RuntimeError)):
            bundle_hypervectors([hv1, hv2])
    
    @pytest.mark.unit
    def test_mismatched_dimensions_bind(self):
        """Test binding hypervectors with mismatched dimensions."""
        hv1 = random_hypervector(1000)
        hv2 = random_hypervector(2000)
        
        with pytest.raises((ValueError, RuntimeError)):
            bind_hypervectors(hv1, hv2)
    
    @pytest.mark.unit
    def test_invalid_dimension(self):
        """Test creating hypervector with invalid dimension."""
        with pytest.raises((ValueError, RuntimeError)):
            random_hypervector(-1)
        
        with pytest.raises((ValueError, RuntimeError)):
            random_hypervector(0)
    
    @pytest.mark.unit
    def test_invalid_sparsity(self):
        """Test creating hypervector with invalid sparsity."""
        with pytest.raises((ValueError, RuntimeError)):
            random_hypervector(1000, sparsity=-0.1)
        
        with pytest.raises((ValueError, RuntimeError)):
            random_hypervector(1000, sparsity=1.1)
    
    @pytest.mark.unit
    def test_invalid_encoding(self):
        """Test creating hypervector with invalid encoding."""
        with pytest.raises((ValueError, RuntimeError)):
            random_hypervector(1000, encoding="invalid")


class TestMemoryEfficiency:
    """Test memory efficiency of operations."""
    
    @pytest.mark.unit
    @pytest.mark.memory
    def test_memory_usage_large_vectors(self):
        """Test memory usage with large hypervectors."""
        # This test would verify memory usage stays within bounds
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large hypervectors
        large_hvs = [random_hypervector(32000) for _ in range(100)]
        
        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
        
        # Verify memory usage is reasonable (less than 500MB for this test)
        assert memory_increase < 500
    
    @pytest.mark.unit
    @pytest.mark.memory
    def test_no_memory_leaks(self):
        """Test that operations don't cause memory leaks."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Perform many operations
        for _ in range(100):
            hv1 = random_hypervector(10000)
            hv2 = random_hypervector(10000)
            _ = bundle_hypervectors([hv1, hv2])
            _ = bind_hypervectors(hv1, hv2)
            del hv1, hv2
            gc.collect()
        
        # Memory should stabilize
        final_memory = process.memory_info().rss
        # This is a basic check - more sophisticated leak detection would be better
        assert final_memory < 1000 * 1024 * 1024  # Less than 1GB