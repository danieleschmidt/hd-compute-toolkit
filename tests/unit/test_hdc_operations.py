"""Unit tests for core HDC operations."""

import numpy as np
import pytest

from tests.conftest import assert_hypervector_properties, assert_similarity_in_range


class TestHDCOperations:
    """Test core HDC operations."""
    
    def test_random_hypervector_generation(self, mock_hdc_backend, medium_dimension):
        """Test random hypervector generation."""
        hv = mock_hdc_backend.random_hv()
        
        assert_hypervector_properties(hv, medium_dimension)
        
        # Check that it's roughly balanced (not all 0s or 1s)
        ones_ratio = hv.sum() / len(hv)
        assert 0.3 < ones_ratio < 0.7, f"Hypervector not balanced: {ones_ratio:.3f}"
    
    def test_random_hypervectors_are_different(self, mock_hdc_backend):
        """Test that random hypervectors are different from each other."""
        hv1 = mock_hdc_backend.random_hv()
        hv2 = mock_hdc_backend.random_hv()
        
        # They should be different (very low probability of being identical)
        assert not np.array_equal(hv1, hv2), "Random hypervectors should be different"
        
        # Hamming distance should be roughly half the dimension
        hamming_distance = np.logical_xor(hv1, hv2).sum()
        expected_distance = len(hv1) // 2
        
        # Allow 20% variance from expected
        assert abs(hamming_distance - expected_distance) < expected_distance * 0.2
    
    def test_bundle_operation(self, mock_hdc_backend):
        """Test bundling operation."""
        hvs = [mock_hdc_backend.random_hv() for _ in range(5)]
        bundled = mock_hdc_backend.bundle(hvs)
        
        assert_hypervector_properties(bundled, mock_hdc_backend.dim)
        
        # Bundled vector should be similar to components
        for hv in hvs:
            similarity = mock_hdc_backend.cosine_similarity(bundled, hv)
            assert similarity > 0.3, f"Bundled vector not similar enough to component: {similarity}"
    
    def test_bundle_single_hypervector(self, mock_hdc_backend):
        """Test bundling with single hypervector."""
        hv = mock_hdc_backend.random_hv()
        bundled = mock_hdc_backend.bundle([hv])
        
        # Should be identical to original
        assert np.array_equal(bundled, hv), "Bundling single vector should return identical vector"
    
    def test_bundle_empty_list(self, mock_hdc_backend):
        """Test bundling empty list raises error."""
        with pytest.raises(ValueError, match="Cannot bundle empty list"):
            mock_hdc_backend.bundle([])
    
    def test_bind_operation(self, mock_hdc_backend):
        """Test binding operation."""
        hv1 = mock_hdc_backend.random_hv()
        hv2 = mock_hdc_backend.random_hv()
        bound = mock_hdc_backend.bind(hv1, hv2)
        
        assert_hypervector_properties(bound, mock_hdc_backend.dim)
        
        # Bound vector should be dissimilar to both inputs
        sim1 = mock_hdc_backend.cosine_similarity(bound, hv1)
        sim2 = mock_hdc_backend.cosine_similarity(bound, hv2)
        
        assert abs(sim1) < 0.3, f"Bound vector too similar to first input: {sim1}"
        assert abs(sim2) < 0.3, f"Bound vector too similar to second input: {sim2}"
    
    def test_bind_commutativity(self, mock_hdc_backend):
        """Test that binding is commutative."""
        hv1 = mock_hdc_backend.random_hv()
        hv2 = mock_hdc_backend.random_hv()
        
        bound1 = mock_hdc_backend.bind(hv1, hv2)
        bound2 = mock_hdc_backend.bind(hv2, hv1)
        
        assert np.array_equal(bound1, bound2), "Binding should be commutative"
    
    def test_bind_inverse_property(self, mock_hdc_backend):
        """Test that binding has inverse property."""
        hv1 = mock_hdc_backend.random_hv()
        hv2 = mock_hdc_backend.random_hv()
        
        bound = mock_hdc_backend.bind(hv1, hv2)
        unbound = mock_hdc_backend.bind(bound, hv2)  # Should recover hv1
        
        similarity = mock_hdc_backend.cosine_similarity(unbound, hv1)
        assert similarity > 0.8, f"Unbinding should recover original vector: {similarity}"
    
    def test_cosine_similarity(self, mock_hdc_backend):
        """Test cosine similarity computation."""
        hv1 = mock_hdc_backend.random_hv()
        hv2 = mock_hdc_backend.random_hv()
        
        # Self-similarity should be 1.0
        self_sim = mock_hdc_backend.cosine_similarity(hv1, hv1)
        assert abs(self_sim - 1.0) < 1e-10, f"Self-similarity should be 1.0: {self_sim}"
        
        # Cross-similarity should be in valid range
        cross_sim = mock_hdc_backend.cosine_similarity(hv1, hv2)
        assert_similarity_in_range(cross_sim)
        
        # Should be close to 0 for random vectors
        assert abs(cross_sim) < 0.3, f"Random vectors should have low similarity: {cross_sim}"
    
    def test_similarity_symmetry(self, mock_hdc_backend):
        """Test that similarity is symmetric."""
        hv1 = mock_hdc_backend.random_hv()
        hv2 = mock_hdc_backend.random_hv()
        
        sim1 = mock_hdc_backend.cosine_similarity(hv1, hv2)
        sim2 = mock_hdc_backend.cosine_similarity(hv2, hv1)
        
        assert abs(sim1 - sim2) < 1e-10, "Similarity should be symmetric"


class TestHDCProperties:
    """Test mathematical properties of HDC operations."""
    
    def test_bundle_distributivity(self, mock_hdc_backend):
        """Test distributivity properties of bundling."""
        hv1 = mock_hdc_backend.random_hv()
        hv2 = mock_hdc_backend.random_hv()
        hv3 = mock_hdc_backend.random_hv()
        
        # Bundle should be associative-like
        bundle_12 = mock_hdc_backend.bundle([hv1, hv2])
        result1 = mock_hdc_backend.bundle([bundle_12, hv3])
        
        bundle_23 = mock_hdc_backend.bundle([hv2, hv3])
        result2 = mock_hdc_backend.bundle([hv1, bundle_23])
        
        # Results should be similar (exact equality not expected due to majority vote)
        similarity = mock_hdc_backend.cosine_similarity(result1, result2)
        assert similarity > 0.8, f"Bundle operations should be associative-like: {similarity}"
    
    def test_bind_associativity(self, mock_hdc_backend):
        """Test associativity of binding."""
        hv1 = mock_hdc_backend.random_hv()
        hv2 = mock_hdc_backend.random_hv()
        hv3 = mock_hdc_backend.random_hv()
        
        # (hv1 ⊛ hv2) ⊛ hv3 = hv1 ⊛ (hv2 ⊛ hv3)
        result1 = mock_hdc_backend.bind(mock_hdc_backend.bind(hv1, hv2), hv3)
        result2 = mock_hdc_backend.bind(hv1, mock_hdc_backend.bind(hv2, hv3))
        
        assert np.array_equal(result1, result2), "Binding should be associative"
    
    @pytest.mark.parametrize("dimension", [100, 1000, 5000])
    def test_dimension_scaling(self, dimension, device):
        """Test operations work correctly across different dimensions."""
        from tests.conftest import MockHDCBackend
        
        backend = MockHDCBackend(dim=dimension, device=device)
        
        hv1 = backend.random_hv()
        hv2 = backend.random_hv()
        
        # Test all operations work
        bundled = backend.bundle([hv1, hv2])
        bound = backend.bind(hv1, hv2)
        similarity = backend.cosine_similarity(hv1, hv2)
        
        assert_hypervector_properties(bundled, dimension)
        assert_hypervector_properties(bound, dimension)
        assert_similarity_in_range(similarity)


class TestHDCEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_dimension(self):
        """Test that zero dimension raises appropriate error."""
        from tests.conftest import MockHDCBackend
        
        with pytest.raises((ValueError, AssertionError)):
            MockHDCBackend(dim=0)
    
    def test_negative_dimension(self):
        """Test that negative dimension raises appropriate error."""
        from tests.conftest import MockHDCBackend
        
        with pytest.raises((ValueError, AssertionError)):
            MockHDCBackend(dim=-1)
    
    def test_similarity_with_mismatched_dimensions(self, mock_hdc_backend):
        """Test similarity computation with mismatched dimensions."""
        from tests.conftest import MockHDCBackend
        
        other_backend = MockHDCBackend(dim=mock_hdc_backend.dim // 2)
        
        hv1 = mock_hdc_backend.random_hv()
        hv2 = other_backend.random_hv()
        
        with pytest.raises((ValueError, AssertionError)):
            mock_hdc_backend.cosine_similarity(hv1, hv2)
    
    def test_bundle_with_mismatched_dimensions(self, mock_hdc_backend):
        """Test bundling with mismatched dimensions."""
        from tests.conftest import MockHDCBackend
        
        other_backend = MockHDCBackend(dim=mock_hdc_backend.dim // 2)
        
        hv1 = mock_hdc_backend.random_hv()
        hv2 = other_backend.random_hv()
        
        with pytest.raises((ValueError, AssertionError)):
            mock_hdc_backend.bundle([hv1, hv2])