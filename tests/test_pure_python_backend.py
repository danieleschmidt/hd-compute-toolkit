"""Comprehensive tests for the Pure Python HDC backend."""

import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hd_compute.pure_python import HDComputePython
from hd_compute.pure_python.hdc_python import SimpleArray
from hd_compute.utils.validation import HDCValidationError, InvalidParameterError


class TestSimpleArray:
    """Test SimpleArray functionality."""
    
    def test_creation_from_list(self):
        """Test creating SimpleArray from list."""
        data = [1.0, 0.0, 1.0, 0.0]
        arr = SimpleArray(data)
        
        assert len(arr) == 4
        assert arr.shape == (4,)
        assert arr.tolist() == data
    
    def test_creation_from_single_value(self):
        """Test creating SimpleArray from single value."""
        arr = SimpleArray(5.0)
        
        assert len(arr) == 1
        assert arr.shape == (1,)
        assert arr[0] == 5.0
    
    def test_indexing(self):
        """Test array indexing."""
        data = [1.0, 2.0, 3.0]
        arr = SimpleArray(data)
        
        assert arr[0] == 1.0
        assert arr[1] == 2.0
        assert arr[2] == 3.0
        
        arr[1] = 5.0
        assert arr[1] == 5.0
    
    def test_type_conversion(self):
        """Test type conversion."""
        data = [1.0, 0.0, 1.0]
        arr = SimpleArray(data)
        
        bool_arr = arr.astype(bool)
        assert bool_arr[0] is True
        assert bool_arr[1] is False
        assert bool_arr[2] is True
    
    def test_mathematical_operations(self):
        """Test mathematical operations."""
        data1 = [1.0, 2.0, 3.0]
        data2 = [4.0, 5.0, 6.0]
        
        arr1 = SimpleArray(data1)
        arr2 = SimpleArray(data2)
        
        # Test dot product
        dot = arr1.dot(arr2)
        expected = 1*4 + 2*5 + 3*6  # 32
        assert dot == expected
        
        # Test norm
        norm1 = arr1.norm()
        expected_norm = (1*1 + 2*2 + 3*3) ** 0.5  # sqrt(14)
        assert abs(norm1 - expected_norm) < 1e-6
        
        # Test sum
        assert arr1.sum() == 6.0


class TestHDComputePython:
    """Test HDComputePython backend functionality."""
    
    def test_initialization_valid(self):
        """Test valid initialization."""
        hdc = HDComputePython(dim=1000)
        assert hdc.dim == 1000
        assert hdc.device == 'cpu'
    
    def test_initialization_invalid_dimension(self):
        """Test initialization with invalid dimension."""
        with pytest.raises((InvalidParameterError, ValueError)):
            HDComputePython(dim=0)
        
        with pytest.raises((InvalidParameterError, ValueError)):
            HDComputePython(dim=-1)
    
    def test_random_hypervector_generation(self):
        """Test random hypervector generation."""
        hdc = HDComputePython(dim=100)
        
        # Test basic generation
        hv = hdc.random_hv()
        assert len(hv.data) == 100
        assert all(x in [0.0, 1.0] for x in hv.data)
        
        # Test different sparsity levels
        hv_sparse = hdc.random_hv(sparsity=0.2)
        hv_dense = hdc.random_hv(sparsity=0.8)
        
        sparse_ones = sum(hv_sparse.data)
        dense_ones = sum(hv_dense.data)
        
        # Sparse should have fewer 1s than dense (statistically)
        assert sparse_ones < dense_ones
    
    def test_random_hypervector_invalid_sparsity(self):
        """Test random hypervector generation with invalid sparsity."""
        hdc = HDComputePython(dim=100)
        
        with pytest.raises((InvalidParameterError, ValueError)):
            hdc.random_hv(sparsity=-0.1)
        
        with pytest.raises((InvalidParameterError, ValueError)):
            hdc.random_hv(sparsity=1.1)
    
    def test_bundle_operation(self):
        """Test hypervector bundling."""
        hdc = HDComputePython(dim=100)
        
        # Create test hypervectors
        hv1 = hdc.random_hv(sparsity=0.5)
        hv2 = hdc.random_hv(sparsity=0.5)
        hv3 = hdc.random_hv(sparsity=0.5)
        
        # Bundle them
        bundled = hdc.bundle([hv1, hv2, hv3])
        
        assert len(bundled.data) == 100
        assert all(x in [0.0, 1.0] for x in bundled.data)
        
        # Bundled vector should be similar to inputs (majority voting)
        sim1 = hdc.cosine_similarity(bundled, hv1)
        sim2 = hdc.cosine_similarity(bundled, hv2)
        sim3 = hdc.cosine_similarity(bundled, hv3)
        
        # All similarities should be reasonably high
        assert sim1 > 0.3
        assert sim2 > 0.3
        assert sim3 > 0.3
    
    def test_bundle_empty_list(self):
        """Test bundling empty list."""
        hdc = HDComputePython(dim=100)
        
        with pytest.raises((InvalidParameterError, ValueError)):
            hdc.bundle([])
    
    def test_bind_operation(self):
        """Test hypervector binding."""
        hdc = HDComputePython(dim=100)
        
        hv1 = hdc.random_hv(sparsity=0.5)
        hv2 = hdc.random_hv(sparsity=0.5)
        
        # Bind them
        bound = hdc.bind(hv1, hv2)
        
        assert len(bound.data) == 100
        assert all(x in [0.0, 1.0] for x in bound.data)
        
        # Bound vector should be dissimilar to inputs
        sim1 = hdc.cosine_similarity(bound, hv1)
        sim2 = hdc.cosine_similarity(bound, hv2)
        
        # Similarities should be around chance level for XOR
        assert 0.2 < sim1 < 0.8
        assert 0.2 < sim2 < 0.8
        
        # Test binding is reversible (A XOR B XOR B = A)
        unbound = hdc.bind(bound, hv2)
        similarity_to_original = hdc.cosine_similarity(unbound, hv1)
        assert similarity_to_original > 0.8
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        hdc = HDComputePython(dim=100)
        
        hv1 = hdc.random_hv(sparsity=0.5)
        hv2 = hdc.random_hv(sparsity=0.5)
        
        # Similarity with self should be 1.0
        self_sim = hdc.cosine_similarity(hv1, hv1)
        assert abs(self_sim - 1.0) < 1e-6
        
        # Similarity between random vectors should be around chance
        sim = hdc.cosine_similarity(hv1, hv2)
        assert 0.0 < sim < 1.0
        
        # Similarity should be symmetric
        sim_reverse = hdc.cosine_similarity(hv2, hv1)
        assert abs(sim - sim_reverse) < 1e-6
    
    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        hdc = HDComputePython(dim=100)
        
        hv1 = hdc.random_hv(sparsity=0.5)
        hv2 = hdc.random_hv(sparsity=0.5)
        
        # Distance with self should be 0
        self_dist = hdc.hamming_distance(hv1, hv1)
        assert self_dist == 0
        
        # Distance should be symmetric
        dist = hdc.hamming_distance(hv1, hv2)
        dist_reverse = hdc.hamming_distance(hv2, hv1)
        assert dist == dist_reverse
        
        # Distance should be reasonable for random vectors
        assert 0 < dist < 100
    
    def test_permutation(self):
        """Test hypervector permutation."""
        hdc = HDComputePython(dim=10)
        
        # Create a simple pattern
        hv = SimpleArray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        
        # Permute by 2 positions
        permuted = hdc.permute(hv, 2)
        
        # Check that values are shifted correctly
        expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0][-2:] + \
                  [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0][:-2]
        
        for i, expected_val in enumerate(expected):
            assert permuted[i] == expected_val
    
    def test_batch_cosine_similarity(self):
        """Test batch similarity calculation."""
        hdc = HDComputePython(dim=50)
        
        hvs1 = [hdc.random_hv() for _ in range(3)]
        hvs2 = [hdc.random_hv() for _ in range(3)]
        
        batch_sims = hdc.batch_cosine_similarity(hvs1, hvs2)
        
        assert len(batch_sims) == 3
        
        # Compare with individual calculations
        for i in range(3):
            individual_sim = hdc.cosine_similarity(hvs1[i], hvs2[i])
            assert abs(batch_sims[i] - individual_sim) < 1e-6
    
    def test_cleanup_operation(self):
        """Test cleanup/nearest neighbor operation."""
        hdc = HDComputePython(dim=100)
        
        # Create item memory
        items = [hdc.random_hv() for _ in range(5)]
        
        # Add noise to first item
        noisy_item = SimpleArray([
            items[0][i] if i % 10 != 0 else (1.0 - items[0][i])
            for i in range(len(items[0].data))
        ])
        
        # Cleanup should return closest item
        cleaned = hdc.cleanup(noisy_item, items, k=1)
        
        # Check that cleaned item is one of the originals
        similarities = [hdc.cosine_similarity(cleaned, item) for item in items]
        max_sim = max(similarities)
        assert max_sim > 0.9  # Should be very similar to original
    
    def test_sequence_encoding(self):
        """Test sequence encoding."""
        hdc = HDComputePython(dim=100)
        
        # Create sequence elements
        elements = [hdc.random_hv() for _ in range(3)]
        
        # Encode sequence
        encoded = hdc.encode_sequence(elements)
        
        assert len(encoded.data) == 100
        assert all(x in [0.0, 1.0] for x in encoded.data)
        
        # Encoded sequence should be different from individual elements
        for element in elements:
            sim = hdc.cosine_similarity(encoded, element)
            assert sim < 0.8  # Should be dissimilar due to position binding
    
    def test_create_item_memory(self):
        """Test item memory creation."""
        hdc = HDComputePython(dim=100)
        
        items = ['apple', 'banana', 'cherry']
        memory, item_to_index = hdc.create_item_memory(items)
        
        assert len(memory) == 3
        assert len(item_to_index) == 3
        
        for i, item in enumerate(items):
            assert item_to_index[item] == i
            assert len(memory[i].data) == 100


class TestValidationAndErrorHandling:
    """Test validation and error handling."""
    
    def test_dimension_mismatch_in_operations(self):
        """Test dimension mismatch handling."""
        hdc1 = HDComputePython(dim=100)
        hdc2 = HDComputePython(dim=200)
        
        hv1 = hdc1.random_hv()
        hv2 = hdc2.random_hv()
        
        # These operations should fail due to dimension mismatch
        with pytest.raises((ValueError, InvalidParameterError)):
            hdc1.bind(hv1, hv2)
        
        with pytest.raises((ValueError, InvalidParameterError)):
            hdc1.cosine_similarity(hv1, hv2)
    
    def test_invalid_device_specification(self):
        """Test invalid device handling."""
        # Should issue warning but not fail
        hdc = HDComputePython(dim=100, device='gpu')
        assert hdc.device == 'gpu'  # Stored but not used


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_dimension(self):
        """Test minimum dimension."""
        # Very small dimension should work but may issue warning
        hdc = HDComputePython(dim=10)
        hv = hdc.random_hv()
        assert len(hv.data) == 10
    
    def test_extreme_sparsity_levels(self):
        """Test extreme sparsity levels."""
        hdc = HDComputePython(dim=100)
        
        # Very sparse
        hv_sparse = hdc.random_hv(sparsity=0.01)
        ones_count = sum(hv_sparse.data)
        assert ones_count < 10  # Should have very few 1s
        
        # Very dense  
        hv_dense = hdc.random_hv(sparsity=0.99)
        ones_count = sum(hv_dense.data)
        assert ones_count > 90  # Should have many 1s
    
    def test_single_hypervector_bundle(self):
        """Test bundling single hypervector."""
        hdc = HDComputePython(dim=100)
        hv = hdc.random_hv()
        
        # Bundle should return identical hypervector
        bundled = hdc.bundle([hv])
        similarity = hdc.cosine_similarity(hv, bundled)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_large_batch_similarity(self):
        """Test batch operations with empty or mismatched sizes."""
        hdc = HDComputePython(dim=50)
        
        hvs1 = [hdc.random_hv() for _ in range(3)]
        hvs2 = [hdc.random_hv() for _ in range(2)]  # Different size
        
        with pytest.raises(ValueError):
            hdc.batch_cosine_similarity(hvs1, hvs2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])