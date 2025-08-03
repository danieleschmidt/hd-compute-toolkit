"""Tests for caching functionality."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from hd_compute.cache import CacheManager, HypervectorCache


class TestCacheManager:
    """Test CacheManager functionality."""
    
    def test_initialization(self, test_cache_manager):
        """Test cache manager initialization."""
        assert test_cache_manager.cache_dir.exists()
        assert test_cache_manager.max_size_mb == 10
        assert len(test_cache_manager._memory_cache) == 0
    
    def test_basic_get_set(self, test_cache_manager):
        """Test basic get/set operations."""
        key = "test_key"
        value = {"data": "test_value", "number": 42}
        
        # Set value
        test_cache_manager.set(key, value)
        
        # Get value
        retrieved = test_cache_manager.get(key)
        assert retrieved == value
    
    def test_get_nonexistent_key(self, test_cache_manager):
        """Test getting non-existent key returns None."""
        result = test_cache_manager.get("nonexistent_key")
        assert result is None
    
    def test_namespaces(self, test_cache_manager):
        """Test namespace separation."""
        key = "shared_key"
        value1 = "value_in_namespace1"
        value2 = "value_in_namespace2"
        
        # Set same key in different namespaces
        test_cache_manager.set(key, value1, namespace="ns1")
        test_cache_manager.set(key, value2, namespace="ns2")
        
        # Retrieve from different namespaces
        retrieved1 = test_cache_manager.get(key, namespace="ns1")
        retrieved2 = test_cache_manager.get(key, namespace="ns2")
        
        assert retrieved1 == value1
        assert retrieved2 == value2
        assert retrieved1 != retrieved2
    
    def test_delete(self, test_cache_manager):
        """Test deleting cache entries."""
        key = "test_key"
        value = "test_value"
        
        test_cache_manager.set(key, value)
        assert test_cache_manager.get(key) == value
        
        # Delete and verify
        deleted = test_cache_manager.delete(key)
        assert deleted is True
        assert test_cache_manager.get(key) is None
        
        # Delete non-existent key
        deleted = test_cache_manager.delete("nonexistent")
        assert deleted is False
    
    def test_clear_all(self, test_cache_manager):
        """Test clearing all cache entries."""
        # Add multiple entries
        test_cache_manager.set("key1", "value1")
        test_cache_manager.set("key2", "value2", namespace="ns1")
        test_cache_manager.set("key3", "value3", namespace="ns2")
        
        # Clear all
        test_cache_manager.clear()
        
        # All should be gone
        assert test_cache_manager.get("key1") is None
        assert test_cache_manager.get("key2", namespace="ns1") is None
        assert test_cache_manager.get("key3", namespace="ns2") is None
    
    def test_clear_namespace(self, test_cache_manager):
        """Test clearing specific namespace."""
        # Add entries in different namespaces
        test_cache_manager.set("key1", "value1", namespace="ns1")
        test_cache_manager.set("key2", "value2", namespace="ns1")
        test_cache_manager.set("key3", "value3", namespace="ns2")
        
        # Clear only ns1
        test_cache_manager.clear(namespace="ns1")
        
        # ns2 should still exist
        assert test_cache_manager.get("key3", namespace="ns2") == "value3"
    
    def test_memory_cache_eviction(self, temp_dir):
        """Test memory cache eviction when limit is reached."""
        # Create cache with small memory limit
        cache = CacheManager(cache_dir=str(temp_dir / "cache"), max_size_mb=1)
        cache._max_memory_items = 3  # Small limit for testing
        
        # Add items beyond limit
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Memory cache should be limited
        assert len(cache._memory_cache) <= cache._max_memory_items
    
    def test_cache_stats(self, test_cache_manager):
        """Test getting cache statistics."""
        # Add some data
        test_cache_manager.set("key1", "value1")
        test_cache_manager.set("key2", {"large": "data" * 100})
        
        stats = test_cache_manager.get_cache_stats()
        
        assert "memory_cache_size" in stats
        assert "file_cache_size" in stats
        assert "total_size_bytes" in stats
        assert "total_size_mb" in stats
        assert "cache_dir" in stats
        
        assert stats["memory_cache_size"] >= 2
        assert stats["total_size_bytes"] > 0
    
    def test_cached_decorator(self, test_cache_manager):
        """Test function caching decorator."""
        call_count = 0
        
        @test_cache_manager.cached(namespace="functions")
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x * y + call_count  # Include call_count to detect caching
        
        # First call
        result1 = expensive_function(3, 4)
        assert call_count == 1
        
        # Second call with same args - should be cached
        result2 = expensive_function(3, 4)
        assert call_count == 1  # Should not increment
        assert result1 == result2
        
        # Call with different args - should not be cached
        result3 = expensive_function(5, 6)
        assert call_count == 2  # Should increment
        assert result3 != result1
    
    def test_file_persistence(self, test_cache_manager):
        """Test that cache persists to files."""
        key = "persistent_key"
        value = {"data": "persistent_value"}
        
        # Set value
        test_cache_manager.set(key, value)
        
        # Create new cache manager with same directory
        new_cache = CacheManager(
            cache_dir=str(test_cache_manager.cache_dir), 
            max_size_mb=10
        )
        
        # Should be able to retrieve from file
        retrieved = new_cache.get(key)
        assert retrieved == value


class TestHypervectorCache:
    """Test HypervectorCache functionality."""
    
    @pytest.fixture
    def hv_cache(self, test_cache_manager):
        """Create HypervectorCache instance."""
        return HypervectorCache(test_cache_manager)
    
    @pytest.fixture
    def test_hypervector(self):
        """Create test hypervector."""
        np.random.seed(42)
        return np.random.choice([0, 1], size=1000, p=[0.5, 0.5]).astype(bool)
    
    def test_store_and_retrieve_hypervector(self, hv_cache, test_hypervector):
        """Test storing and retrieving hypervectors."""
        dimension = 1000
        sparsity = 0.5
        seed = 42
        
        # Store hypervector
        cache_key = hv_cache.store_hypervector(
            test_hypervector, dimension, sparsity, seed
        )
        
        assert cache_key is not None
        assert len(cache_key) > 0
        
        # Retrieve hypervector
        retrieved_hv = hv_cache.get_hypervector(dimension, sparsity, seed)
        
        assert retrieved_hv is not None
        assert retrieved_hv.shape == test_hypervector.shape
        assert retrieved_hv.dtype == test_hypervector.dtype
        np.testing.assert_array_equal(retrieved_hv, test_hypervector)
    
    def test_cache_key_uniqueness(self, hv_cache):
        """Test that different parameters produce different cache keys."""
        hv = np.random.choice([0, 1], size=1000, p=[0.5, 0.5]).astype(bool)
        
        key1 = hv_cache._generate_hypervector_key(1000, 0.5, 42)
        key2 = hv_cache._generate_hypervector_key(1000, 0.5, 43)  # Different seed
        key3 = hv_cache._generate_hypervector_key(1000, 0.6, 42)  # Different sparsity
        key4 = hv_cache._generate_hypervector_key(2000, 0.5, 42)  # Different dimension
        
        assert key1 != key2
        assert key1 != key3
        assert key1 != key4
        assert key2 != key3
    
    def test_cache_miss(self, hv_cache):
        """Test cache miss returns None."""
        retrieved = hv_cache.get_hypervector(
            dimension=1000, sparsity=0.5, seed=999
        )
        assert retrieved is None
    
    def test_dimension_validation(self, hv_cache, test_hypervector):
        """Test dimension validation during storage."""
        with pytest.raises(ValueError, match="Hypervector dimension.*!= expected"):
            hv_cache.store_hypervector(
                test_hypervector, 
                dimension=500,  # Wrong dimension
                sparsity=0.5, 
                seed=42
            )
    
    def test_metadata_in_cache_key(self, hv_cache, test_hypervector):
        """Test that metadata affects cache key."""
        metadata1 = {"type": "random", "version": 1}
        metadata2 = {"type": "random", "version": 2}
        
        key1 = hv_cache._generate_hypervector_key(1000, 0.5, 42, metadata1)
        key2 = hv_cache._generate_hypervector_key(1000, 0.5, 42, metadata2)
        
        assert key1 != key2
    
    def test_compression_decompression(self, hv_cache):
        """Test hypervector compression and decompression."""
        # Create test binary hypervector
        hv = np.random.choice([0, 1], size=1000, p=[0.5, 0.5]).astype(bool)
        
        # Compress
        compressed = hv_cache._compress_hypervector(hv)
        
        # Should be smaller (8:1 ratio for boolean to bytes)
        assert len(compressed) <= len(hv) // 8 + 1
        
        # Decompress
        decompressed = hv_cache._decompress_hypervector(
            compressed, len(hv), str(hv.dtype)
        )
        
        # Should match original
        assert decompressed.shape == hv.shape
        np.testing.assert_array_equal(decompressed, hv)
    
    def test_cache_statistics(self, hv_cache, test_hypervector):
        """Test cache statistics tracking."""
        # Initially no hits/misses
        stats = hv_cache.get_cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['stores'] == 0
        
        # Store hypervector
        hv_cache.store_hypervector(test_hypervector, 1000, 0.5, 42)
        stats = hv_cache.get_cache_stats()
        assert stats['stores'] == 1
        
        # Cache hit
        retrieved = hv_cache.get_hypervector(1000, 0.5, 42)
        assert retrieved is not None
        stats = hv_cache.get_cache_stats()
        assert stats['hits'] == 1
        
        # Cache miss
        missed = hv_cache.get_hypervector(1000, 0.5, 999)
        assert missed is None
        stats = hv_cache.get_cache_stats()
        assert stats['misses'] == 1
        
        # Check hit rate
        assert stats['hit_rate'] == 0.5  # 1 hit out of 2 attempts
    
    def test_clear_cache(self, hv_cache, test_hypervector):
        """Test clearing hypervector cache."""
        # Store hypervector
        hv_cache.store_hypervector(test_hypervector, 1000, 0.5, 42)
        
        # Verify it's there
        retrieved = hv_cache.get_hypervector(1000, 0.5, 42)
        assert retrieved is not None
        
        # Clear cache
        hv_cache.clear_cache()
        
        # Should be gone
        retrieved = hv_cache.get_hypervector(1000, 0.5, 42)
        assert retrieved is None
        
        # Stats should be reset
        stats = hv_cache.get_cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['stores'] == 0
    
    def test_get_cached_or_generate(self, hv_cache, mock_hdc_backend):
        """Test getting cached hypervector or generating new one."""
        dimension = 1000
        sparsity = 0.5
        seed = 42
        
        # First call should generate and cache
        hv1 = hv_cache.get_cached_hypervector_or_generate(
            mock_hdc_backend, dimension, sparsity, seed
        )
        assert hv1 is not None
        assert hv1.shape == (dimension,)
        
        # Second call should retrieve from cache
        hv2 = hv_cache.get_cached_hypervector_or_generate(
            mock_hdc_backend, dimension, sparsity, seed
        )
        
        # Should be identical (from cache)
        np.testing.assert_array_equal(hv1, hv2)
        
        # Verify cache was used
        stats = hv_cache.get_cache_stats()
        assert stats['hits'] >= 1
    
    def test_memory_usage_estimate(self, hv_cache, test_hypervector):
        """Test memory usage estimation."""
        # Store some hypervectors
        for i in range(3):
            hv_cache.store_hypervector(test_hypervector, 1000, 0.5, i)
        
        usage = hv_cache.get_memory_usage_estimate()
        
        assert 'estimated_hypervectors_count' in usage
        assert 'estimated_total_memory_mb' in usage
        assert 'cache_file_size_mb' in usage
        
        assert usage['estimated_hypervectors_count'] >= 3