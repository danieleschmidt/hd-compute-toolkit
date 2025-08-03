"""Tests for PyTorch HDC backend implementation."""

import pytest
import torch
import numpy as np

from tests.conftest import assert_hypervector_properties, assert_similarity_in_range


class TestHDComputeTorch:
    """Test PyTorch HDC backend functionality."""
    
    def test_initialization(self, medium_dimension):
        """Test HDComputeTorch initialization."""
        from hd_compute import HDComputeTorch
        
        hdc = HDComputeTorch(dim=medium_dimension, device='cpu')
        assert hdc.dim == medium_dimension
        assert hdc.device == 'cpu'
        assert hdc.dtype == torch.float32
    
    def test_random_hv_generation(self, pytorch_backend, medium_dimension):
        """Test random hypervector generation."""
        hv = pytorch_backend.random_hv()
        
        assert isinstance(hv, torch.Tensor)
        assert hv.shape == (medium_dimension,)
        assert hv.dtype == pytorch_backend.dtype
        
        # Check sparsity is approximately correct
        sparsity = torch.mean(hv.float()).item()
        assert 0.4 <= sparsity <= 0.6, f"Sparsity {sparsity} not around 0.5"
    
    def test_random_hv_batch(self, pytorch_backend, medium_dimension):
        """Test batch random hypervector generation."""
        batch_size = 5
        hvs = pytorch_backend.random_hv(batch_size=batch_size)
        
        assert isinstance(hvs, torch.Tensor)
        assert hvs.shape == (batch_size, medium_dimension)
        
        # Check that vectors are different
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                hamming_dist = torch.sum(torch.logical_xor(hvs[i].bool(), hvs[j].bool())).item()
                assert hamming_dist > medium_dimension * 0.3, "Vectors too similar"
    
    def test_bundle_operation(self, pytorch_backend):
        """Test bundling operation."""
        hvs = [pytorch_backend.random_hv() for _ in range(5)]
        bundled = pytorch_backend.bundle(hvs)
        
        assert isinstance(bundled, torch.Tensor)
        assert bundled.shape == hvs[0].shape
        assert bundled.dtype == pytorch_backend.dtype
    
    def test_bundle_empty_list(self, pytorch_backend):
        """Test bundling with empty list raises error."""
        with pytest.raises(ValueError, match="Cannot bundle empty list"):
            pytorch_backend.bundle([])
    
    def test_bind_operation(self, pytorch_backend):
        """Test binding operation."""
        hv1 = pytorch_backend.random_hv()
        hv2 = pytorch_backend.random_hv()
        
        bound = pytorch_backend.bind(hv1, hv2)
        
        assert isinstance(bound, torch.Tensor)
        assert bound.shape == hv1.shape
        assert bound.dtype == pytorch_backend.dtype
        
        # XOR property: bind(A, A) should be close to zero vector
        self_bound = pytorch_backend.bind(hv1, hv1)
        zero_ratio = torch.mean(self_bound.float()).item()
        assert zero_ratio < 0.1, "Self-binding should produce mostly zeros"
    
    def test_cosine_similarity(self, pytorch_backend):
        """Test cosine similarity computation."""
        hv1 = pytorch_backend.random_hv()
        hv2 = pytorch_backend.random_hv()
        
        # Self-similarity should be 1
        self_sim = pytorch_backend.cosine_similarity(hv1, hv1)
        assert abs(self_sim - 1.0) < 1e-6, f"Self-similarity {self_sim} != 1.0"
        
        # Different vectors should have similarity around 0
        diff_sim = pytorch_backend.cosine_similarity(hv1, hv2)
        assert_similarity_in_range(diff_sim)
        assert abs(diff_sim) < 0.2, f"Random vectors too similar: {diff_sim}"
    
    def test_hamming_distance(self, pytorch_backend, medium_dimension):
        """Test Hamming distance computation."""
        hv1 = pytorch_backend.random_hv()
        hv2 = pytorch_backend.random_hv()
        
        # Self-distance should be 0
        self_dist = pytorch_backend.hamming_distance(hv1, hv1)
        assert self_dist == 0, f"Self-distance {self_dist} != 0"
        
        # Different vectors should have distance around dim/2
        diff_dist = pytorch_backend.hamming_distance(hv1, hv2)
        expected_range = (medium_dimension * 0.3, medium_dimension * 0.7)
        assert expected_range[0] <= diff_dist <= expected_range[1], \
            f"Hamming distance {diff_dist} not in expected range {expected_range}"
    
    def test_permute_operation(self, pytorch_backend, medium_dimension):
        """Test permutation operation."""
        hv = pytorch_backend.random_hv()
        positions = 10
        
        permuted = pytorch_backend.permute(hv, positions)
        
        assert isinstance(permuted, torch.Tensor)
        assert permuted.shape == hv.shape
        
        # Check that permutation actually moved elements
        assert not torch.equal(hv, permuted), "Permutation didn't change vector"
        
        # Check that reverse permutation restores original
        restored = pytorch_backend.permute(permuted, -positions)
        assert torch.equal(hv, restored), "Reverse permutation failed"
    
    def test_batch_cosine_similarity(self, pytorch_backend):
        """Test batch cosine similarity computation."""
        batch_size = 3
        hvs1 = pytorch_backend.random_hv(batch_size=batch_size)
        hvs2 = pytorch_backend.random_hv(batch_size=batch_size)
        
        similarities = pytorch_backend.batch_cosine_similarity(hvs1, hvs2)
        
        assert isinstance(similarities, torch.Tensor)
        assert similarities.shape == (batch_size,)
        
        # All similarities should be in valid range
        for sim in similarities:
            assert_similarity_in_range(sim.item())
    
    def test_cleanup_operation(self, pytorch_backend):
        """Test cleanup operation with item memory."""
        # Create item memory
        clean_hvs = pytorch_backend.random_hv(batch_size=5)
        
        # Create noisy version of first vector
        noise = pytorch_backend.random_hv() * 0.1  # Small noise
        noisy_hv = clean_hvs[0] + noise
        
        cleaned = pytorch_backend.cleanup(noisy_hv, clean_hvs, k=1)
        
        assert isinstance(cleaned, torch.Tensor)
        assert cleaned.shape == noisy_hv.shape
        
        # Cleaned vector should be one from the memory
        assert torch.equal(cleaned, clean_hvs[0]), "Cleanup didn't return expected vector"
    
    def test_encode_sequence(self, pytorch_backend):
        """Test sequence encoding with positional information."""
        sequence = [pytorch_backend.random_hv() for _ in range(3)]
        
        encoded = pytorch_backend.encode_sequence(sequence)
        
        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == sequence[0].shape
        
        # Encoded sequence should be different from individual elements
        for hv in sequence:
            similarity = pytorch_backend.cosine_similarity(encoded, hv)
            assert abs(similarity) < 0.5, "Encoded sequence too similar to individual elements"
    
    def test_create_item_memory(self, pytorch_backend):
        """Test item memory creation."""
        items = ['apple', 'banana', 'cherry']
        
        memory, item_to_index = pytorch_backend.create_item_memory(items)
        
        assert isinstance(memory, torch.Tensor)
        assert memory.shape == (len(items), pytorch_backend.dim)
        assert isinstance(item_to_index, dict)
        assert set(item_to_index.keys()) == set(items)
        assert list(item_to_index.values()) == list(range(len(items)))
    
    def test_device_consistency(self, medium_dimension):
        """Test that operations maintain device consistency."""
        if torch.cuda.is_available():
            from hd_compute import HDComputeTorch
            
            hdc = HDComputeTorch(dim=medium_dimension, device='cuda')
            hv1 = hdc.random_hv()
            hv2 = hdc.random_hv()
            
            assert hv1.device.type == 'cuda'
            assert hv2.device.type == 'cuda'
            
            bound = hdc.bind(hv1, hv2)
            assert bound.device.type == 'cuda'
            
            bundled = hdc.bundle([hv1, hv2])
            assert bundled.device.type == 'cuda'
    
    def test_dtype_consistency(self, pytorch_backend):
        """Test that operations maintain dtype consistency."""
        hv1 = pytorch_backend.random_hv()
        hv2 = pytorch_backend.random_hv()
        
        assert hv1.dtype == pytorch_backend.dtype
        assert hv2.dtype == pytorch_backend.dtype
        
        bound = pytorch_backend.bind(hv1, hv2)
        bundled = pytorch_backend.bundle([hv1, hv2])
        
        assert bound.dtype == pytorch_backend.dtype
        assert bundled.dtype == pytorch_backend.dtype
    
    @pytest.mark.parametrize("sparsity", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_sparsity_control(self, pytorch_backend, sparsity):
        """Test that sparsity parameter controls hypervector density."""
        hv = pytorch_backend.random_hv(sparsity=sparsity)
        actual_sparsity = torch.mean(hv.float()).item()
        
        # Allow some tolerance due to randomness
        assert abs(actual_sparsity - sparsity) < 0.1, \
            f"Actual sparsity {actual_sparsity} too far from target {sparsity}"