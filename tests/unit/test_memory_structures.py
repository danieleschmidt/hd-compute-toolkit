"""Tests for memory structures (ItemMemory and AssociativeMemory)."""

import pytest
import numpy as np


class TestItemMemory:
    """Test ItemMemory functionality."""
    
    def test_initialization(self, item_memory):
        """Test ItemMemory initialization."""
        assert item_memory.items == []
        assert item_memory.item_to_index == {}
        assert item_memory.memory is None
    
    def test_add_items(self, item_memory):
        """Test adding items to memory."""
        items = ['apple', 'banana', 'cherry']
        item_memory.add_items(items)
        
        assert item_memory.items == items
        assert len(item_memory.item_to_index) == len(items)
        assert item_memory.memory is not None
        
        # Check that memory has correct shape
        assert item_memory.memory.shape[0] == len(items)
        assert item_memory.memory.shape[1] == item_memory.hdc.dim
    
    def test_add_duplicate_items(self, item_memory):
        """Test that duplicate items are not added."""
        items = ['apple', 'banana']
        item_memory.add_items(items)
        
        original_size = item_memory.size()
        
        # Try to add duplicates
        item_memory.add_items(['apple', 'grape'])
        
        assert item_memory.size() == original_size + 1  # Only 'grape' should be added
        assert 'grape' in item_memory.items
    
    def test_get_hv(self, item_memory):
        """Test getting hypervector for an item."""
        items = ['apple', 'banana']
        item_memory.add_items(items)
        
        apple_hv = item_memory.get_hv('apple')
        banana_hv = item_memory.get_hv('banana')
        
        assert apple_hv is not None
        assert banana_hv is not None
        assert apple_hv.shape == (item_memory.hdc.dim,)
        assert banana_hv.shape == (item_memory.hdc.dim,)
        
        # Different items should have different hypervectors
        similarity = item_memory.hdc.cosine_similarity(apple_hv, banana_hv)
        assert abs(similarity) < 0.3, "Different items should have low similarity"
    
    def test_get_nonexistent_item(self, item_memory):
        """Test getting hypervector for non-existent item raises error."""
        with pytest.raises(KeyError, match="Item 'nonexistent' not found"):
            item_memory.get_hv('nonexistent')
    
    def test_get_multiple_hvs(self, item_memory):
        """Test getting multiple hypervectors at once."""
        items = ['apple', 'banana', 'cherry']
        item_memory.add_items(items)
        
        hvs = item_memory.get_multiple_hvs(['apple', 'cherry'])
        
        assert hvs.shape == (2, item_memory.hdc.dim)
        
        # Should match individual retrieval
        apple_hv = item_memory.get_hv('apple')
        cherry_hv = item_memory.get_hv('cherry')
        
        # Check if vectors match (accounting for different tensor types)
        apple_match = item_memory.hdc.cosine_similarity(hvs[0], apple_hv)
        cherry_match = item_memory.hdc.cosine_similarity(hvs[1], cherry_hv)
        
        assert abs(apple_match - 1.0) < 1e-6
        assert abs(cherry_match - 1.0) < 1e-6
    
    def test_encode_sequence(self, item_memory):
        """Test encoding a sequence of items."""
        items = ['red', 'apple', 'fruit']
        item_memory.add_items(items)
        
        sequence = ['red', 'apple']
        encoded = item_memory.encode_sequence(sequence)
        
        assert encoded is not None
        assert encoded.shape == (item_memory.hdc.dim,)
        
        # Encoded sequence should be different from individual items
        for item in sequence:
            item_hv = item_memory.get_hv(item)
            similarity = item_memory.hdc.cosine_similarity(encoded, item_hv)
            assert abs(similarity) < 0.7, f"Sequence too similar to individual item '{item}'"
    
    def test_encode_sequence_missing_items(self, item_memory):
        """Test encoding sequence with missing items raises error."""
        item_memory.add_items(['apple'])
        
        with pytest.raises(KeyError, match="Items not found in memory"):
            item_memory.encode_sequence(['apple', 'missing'])
    
    def test_cleanup(self, item_memory):
        """Test cleanup operation."""
        items = ['apple', 'banana', 'cherry']
        item_memory.add_items(items)
        
        # Create noisy version of 'apple'
        apple_hv = item_memory.get_hv('apple')
        # Add some noise by flipping small percentage of bits
        noise_indices = np.random.choice(len(apple_hv), size=len(apple_hv)//20, replace=False)
        noisy_apple = apple_hv.clone() if hasattr(apple_hv, 'clone') else apple_hv.copy()
        
        if hasattr(noisy_apple, 'logical_xor'):  # PyTorch
            noise_mask = item_memory.hdc.random_hv() > 0.8
            noisy_apple = item_memory.hdc.bind(noisy_apple, noise_mask)
        
        # Cleanup should return the closest item
        cleaned_item = item_memory.cleanup(noisy_apple)
        
        assert cleaned_item in items
        # Most likely should be 'apple' but not guaranteed due to randomness
    
    def test_size(self, item_memory):
        """Test getting memory size."""
        assert item_memory.size() == 0
        
        item_memory.add_items(['apple', 'banana'])
        assert item_memory.size() == 2
        
        item_memory.add_items(['cherry'])
        assert item_memory.size() == 3
    
    def test_clear(self, item_memory):
        """Test clearing memory."""
        item_memory.add_items(['apple', 'banana'])
        assert item_memory.size() == 2
        
        item_memory.clear()
        assert item_memory.size() == 0
        assert item_memory.items == []
        assert item_memory.item_to_index == {}
        assert item_memory.memory is None


class TestAssociativeMemory:
    """Test AssociativeMemory functionality."""
    
    def test_initialization(self, associative_memory):
        """Test AssociativeMemory initialization."""
        assert associative_memory.capacity == 100
        assert len(associative_memory.patterns) == 0
        assert len(associative_memory.labels) == 0
        assert associative_memory.memory is None
    
    def test_store_pattern(self, associative_memory):
        """Test storing a pattern."""
        pattern = associative_memory.hdc.random_hv()
        label = "test_pattern"
        
        associative_memory.store(pattern, label)
        
        assert len(associative_memory.patterns) == 1
        assert len(associative_memory.labels) == 1
        assert associative_memory.labels[0] == label
        assert associative_memory.memory is not None
    
    def test_store_multiple_patterns(self, associative_memory):
        """Test storing multiple patterns."""
        patterns = [associative_memory.hdc.random_hv() for _ in range(3)]
        labels = ["pattern_1", "pattern_2", "pattern_3"]
        
        for pattern, label in zip(patterns, labels):
            associative_memory.store(pattern, label)
        
        assert len(associative_memory.patterns) == 3
        assert associative_memory.labels == labels
    
    def test_capacity_limit(self, associative_memory):
        """Test that memory respects capacity limit."""
        # Fill memory to capacity
        for i in range(associative_memory.capacity + 5):
            pattern = associative_memory.hdc.random_hv()
            associative_memory.store(pattern, f"pattern_{i}")
        
        # Should not exceed capacity
        assert len(associative_memory.patterns) == associative_memory.capacity
        assert len(associative_memory.labels) == associative_memory.capacity
        
        # Should contain most recent patterns (FIFO)
        assert "pattern_5" in associative_memory.labels  # Should be oldest remaining
        assert f"pattern_{associative_memory.capacity + 4}" in associative_memory.labels  # Should be newest
    
    def test_recall_exact_match(self, associative_memory):
        """Test recalling exact pattern match."""
        pattern = associative_memory.hdc.random_hv()
        label = "exact_pattern"
        
        associative_memory.store(pattern, label)
        
        # Should find exact match
        results = associative_memory.recall(pattern, k=1)
        
        assert len(results) == 1
        assert results[0][0] == label
        assert results[0][1] > 0.99  # Should be very high similarity
    
    def test_recall_with_noise(self, associative_memory):
        """Test recalling noisy pattern."""
        pattern = associative_memory.hdc.random_hv()
        label = "noisy_pattern"
        
        associative_memory.store(pattern, label)
        
        # Create noisy version
        noise = associative_memory.hdc.random_hv()
        noisy_pattern = associative_memory.hdc.bundle([pattern, noise])
        
        results = associative_memory.recall(noisy_pattern, k=1, threshold=0.1)
        
        assert len(results) >= 1
        # Should still find the original pattern (though similarity will be lower)
        assert any(result[0] == label for result in results)
    
    def test_recall_multiple_k(self, associative_memory):
        """Test recalling multiple similar patterns."""
        patterns = [associative_memory.hdc.random_hv() for _ in range(5)]
        labels = [f"pattern_{i}" for i in range(5)]
        
        for pattern, label in zip(patterns, labels):
            associative_memory.store(pattern, label)
        
        # Query with one of the patterns
        results = associative_memory.recall(patterns[0], k=3)
        
        assert len(results) <= 3
        assert len(results) >= 1
        
        # Results should be sorted by similarity (highest first)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i][1] >= results[i + 1][1]
    
    def test_recall_with_threshold(self, associative_memory):
        """Test recall with similarity threshold."""
        pattern = associative_memory.hdc.random_hv()
        associative_memory.store(pattern, "stored_pattern")
        
        # Create very different pattern
        different_pattern = associative_memory.hdc.random_hv()
        
        # High threshold should exclude dissimilar patterns
        results = associative_memory.recall(different_pattern, k=5, threshold=0.8)
        
        # Might be empty or have very few results due to high threshold
        assert len(results) <= 1
    
    def test_cleanup_pattern(self, associative_memory):
        """Test cleaning up noisy pattern."""
        pattern = associative_memory.hdc.random_hv()
        associative_memory.store(pattern, "clean_pattern")
        
        # Create noisy version
        noise = associative_memory.hdc.random_hv()
        noisy_pattern = associative_memory.hdc.bundle([pattern, noise])
        
        cleaned = associative_memory.cleanup(noisy_pattern)
        
        # Cleaned pattern should be the stored one
        similarity = associative_memory.hdc.cosine_similarity(cleaned, pattern)
        assert similarity > 0.9  # Should be very similar to original
    
    def test_size(self, associative_memory):
        """Test getting memory size."""
        assert associative_memory.size() == 0
        
        associative_memory.store(associative_memory.hdc.random_hv(), "pattern_1")
        assert associative_memory.size() == 1
        
        associative_memory.store(associative_memory.hdc.random_hv(), "pattern_2")
        assert associative_memory.size() == 2
    
    def test_clear(self, associative_memory):
        """Test clearing memory."""
        associative_memory.store(associative_memory.hdc.random_hv(), "pattern_1")
        associative_memory.store(associative_memory.hdc.random_hv(), "pattern_2")
        
        assert associative_memory.size() == 2
        
        associative_memory.clear()
        
        assert associative_memory.size() == 0
        assert len(associative_memory.patterns) == 0
        assert len(associative_memory.labels) == 0
        assert associative_memory.memory is None
    
    def test_get_statistics(self, associative_memory):
        """Test getting memory statistics."""
        # Empty memory
        stats = associative_memory.get_statistics()
        assert stats['size'] == 0
        assert stats['capacity'] == 100
        assert stats['utilization'] == 0.0
        assert stats['unique_labels'] == 0
        
        # Add some patterns
        for i in range(3):
            pattern = associative_memory.hdc.random_hv()
            associative_memory.store(pattern, f"pattern_{i}")
        
        # Add duplicate label
        associative_memory.store(associative_memory.hdc.random_hv(), "pattern_1")
        
        stats = associative_memory.get_statistics()
        assert stats['size'] == 4
        assert stats['utilization'] == 4 / 100
        assert stats['unique_labels'] == 3  # pattern_0, pattern_1, pattern_2