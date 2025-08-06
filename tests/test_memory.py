"""Tests for memory components."""

import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hd_compute.pure_python import HDComputePython
from hd_compute.memory import ItemMemory, AssociativeMemory


class TestItemMemory:
    """Test ItemMemory functionality."""
    
    def test_initialization_empty(self):
        """Test initialization without items."""
        hdc = HDComputePython(dim=100)
        memory = ItemMemory(hdc)
        
        assert memory.size() == 0
        assert len(memory.items) == 0
        assert len(memory.item_to_index) == 0
    
    def test_initialization_with_items(self):
        """Test initialization with items."""
        hdc = HDComputePython(dim=100)
        items = ['cat', 'dog', 'bird']
        memory = ItemMemory(hdc, items)
        
        assert memory.size() == 3
        assert set(memory.items) == set(items)
        
        for i, item in enumerate(items):
            assert memory.item_to_index[item] == i
    
    def test_add_items(self):
        """Test adding new items."""
        hdc = HDComputePython(dim=100)
        memory = ItemMemory(hdc)
        
        # Add initial items
        items1 = ['apple', 'banana']
        memory.add_items(items1)
        
        assert memory.size() == 2
        assert 'apple' in memory.item_to_index
        assert 'banana' in memory.item_to_index
        
        # Add more items
        items2 = ['cherry', 'date']
        memory.add_items(items2)
        
        assert memory.size() == 4
        assert memory.item_to_index['cherry'] == 2
        assert memory.item_to_index['date'] == 3
    
    def test_add_duplicate_items(self):
        """Test adding duplicate items."""
        hdc = HDComputePython(dim=100)
        memory = ItemMemory(hdc, ['apple', 'banana'])
        
        initial_size = memory.size()
        
        # Adding duplicates should not increase size
        memory.add_items(['apple', 'cherry'])
        
        assert memory.size() == initial_size + 1  # Only cherry is new
        assert 'cherry' in memory.item_to_index
    
    def test_get_hypervector(self):
        """Test retrieving hypervectors."""
        hdc = HDComputePython(dim=100)
        items = ['red', 'green', 'blue']
        memory = ItemMemory(hdc, items)
        
        # Get hypervectors
        red_hv = memory.get_hv('red')
        green_hv = memory.get_hv('green')
        blue_hv = memory.get_hv('blue')
        
        assert len(red_hv.data) == 100
        assert len(green_hv.data) == 100
        assert len(blue_hv.data) == 100
        
        # Different items should have different hypervectors
        sim_rg = hdc.cosine_similarity(red_hv, green_hv)
        sim_rb = hdc.cosine_similarity(red_hv, blue_hv)
        sim_gb = hdc.cosine_similarity(green_hv, blue_hv)
        
        # Should be dissimilar (random hypervectors)
        assert sim_rg < 0.8
        assert sim_rb < 0.8
        assert sim_gb < 0.8
    
    def test_get_nonexistent_item(self):
        """Test retrieving non-existent item."""
        hdc = HDComputePython(dim=100)
        memory = ItemMemory(hdc, ['apple'])
        
        with pytest.raises(KeyError):
            memory.get_hv('orange')
    
    def test_get_multiple_hypervectors(self):
        """Test retrieving multiple hypervectors."""
        hdc = HDComputePython(dim=100)
        items = ['x', 'y', 'z']
        memory = ItemMemory(hdc, items)
        
        hvs = memory.get_multiple_hvs(['x', 'z'])
        
        assert len(hvs) == 2
        
        # Compare with individual retrieval
        x_hv = memory.get_hv('x')
        z_hv = memory.get_hv('z')
        
        assert hdc.cosine_similarity(hvs[0], x_hv) == 1.0
        assert hdc.cosine_similarity(hvs[1], z_hv) == 1.0
    
    def test_encode_sequence(self):
        """Test sequence encoding."""
        hdc = HDComputePython(dim=100)
        items = ['the', 'quick', 'brown', 'fox']
        memory = ItemMemory(hdc, items)
        
        sequence = ['the', 'quick', 'fox']
        encoded = memory.encode_sequence(sequence)
        
        assert len(encoded.data) == 100
        
        # Encoded sequence should be different from individual items
        for item in sequence:
            item_hv = memory.get_hv(item)
            sim = hdc.cosine_similarity(encoded, item_hv)
            assert sim < 0.8  # Dissimilar due to positional encoding
    
    def test_encode_sequence_with_missing_items(self):
        """Test sequence encoding with missing items."""
        hdc = HDComputePython(dim=100)
        memory = ItemMemory(hdc, ['a', 'b'])
        
        with pytest.raises(KeyError):
            memory.encode_sequence(['a', 'c'])  # 'c' not in memory
    
    def test_cleanup_operation(self):
        """Test cleanup operation."""
        hdc = HDComputePython(dim=100)
        items = ['cat', 'dog', 'fish']
        memory = ItemMemory(hdc, items)
        
        # Get a clean hypervector
        cat_hv = memory.get_hv('cat')
        
        # Add some noise
        noisy_hv = hdc.bind(cat_hv, hdc.random_hv())
        
        # Cleanup should return most similar item
        cleaned_item = memory.cleanup(noisy_hv)
        
        assert cleaned_item in items
        # Note: Due to randomness, we can't guarantee it returns 'cat'
        # but it should return a valid item
    
    def test_clear_memory(self):
        """Test clearing memory."""
        hdc = HDComputePython(dim=100)
        memory = ItemMemory(hdc, ['a', 'b', 'c'])
        
        assert memory.size() == 3
        
        memory.clear()
        
        assert memory.size() == 0
        assert len(memory.items) == 0
        assert len(memory.item_to_index) == 0


class TestAssociativeMemory:
    """Test AssociativeMemory functionality."""
    
    def test_initialization(self):
        """Test initialization."""
        hdc = HDComputePython(dim=100)
        memory = AssociativeMemory(hdc, capacity=50)
        
        assert memory.capacity == 50
        assert memory.size() == 0
    
    def test_store_and_recall(self):
        """Test storing and recalling patterns."""
        hdc = HDComputePython(dim=100)
        memory = AssociativeMemory(hdc)
        
        # Store some patterns
        pattern1 = hdc.random_hv()
        pattern2 = hdc.random_hv()
        
        memory.store(pattern1, 'pattern1')
        memory.store(pattern2, 'pattern2')
        
        assert memory.size() == 2
        
        # Recall exact patterns
        recalls1 = memory.recall(pattern1, k=1)
        recalls2 = memory.recall(pattern2, k=1)
        
        assert len(recalls1) == 1
        assert len(recalls2) == 1
        assert recalls1[0][0] == 'pattern1'
        assert recalls2[0][0] == 'pattern2'
        assert recalls1[0][1] == 1.0  # Perfect similarity
        assert recalls2[0][1] == 1.0
    
    def test_recall_with_noise(self):
        """Test recall with noisy patterns."""
        hdc = HDComputePython(dim=100)
        memory = AssociativeMemory(hdc)
        
        # Store original pattern
        original = hdc.random_hv()
        memory.store(original, 'original')
        
        # Create noisy version by binding with random vector
        noise = hdc.random_hv()
        noisy = hdc.bind(original, noise)
        
        # Recall should still find the original (though with lower similarity)
        recalls = memory.recall(noisy, k=1, threshold=0.0)
        
        assert len(recalls) >= 1
        assert recalls[0][0] == 'original'
        assert 0.0 < recalls[0][1] < 1.0  # Similarity reduced due to noise
    
    def test_recall_multiple_results(self):
        """Test recalling multiple results."""
        hdc = HDComputePython(dim=100)
        memory = AssociativeMemory(hdc)
        
        # Store multiple patterns
        for i in range(5):
            pattern = hdc.random_hv()
            memory.store(pattern, f'pattern{i}')
        
        # Query with one of the patterns
        query = memory.patterns[0]  # First stored pattern
        recalls = memory.recall(query, k=3)
        
        assert len(recalls) <= 3
        assert recalls[0][0] == 'pattern0'  # Best match should be exact
        assert recalls[0][1] == 1.0
    
    def test_recall_with_threshold(self):
        """Test recall with similarity threshold."""
        hdc = HDComputePython(dim=100)
        memory = AssociativeMemory(hdc)
        
        # Store patterns
        pattern = hdc.random_hv()
        memory.store(pattern, 'pattern')
        
        # Query with very dissimilar pattern
        dissimilar = hdc.random_hv()
        
        # High threshold should return no results
        recalls = memory.recall(dissimilar, threshold=0.9)
        assert len(recalls) == 0
        
        # Low threshold should return results
        recalls = memory.recall(dissimilar, threshold=0.0)
        assert len(recalls) > 0
    
    def test_capacity_limit(self):
        """Test memory capacity limit."""
        hdc = HDComputePython(dim=100)
        memory = AssociativeMemory(hdc, capacity=3)
        
        # Store more patterns than capacity
        patterns = []
        for i in range(5):
            pattern = hdc.random_hv()
            patterns.append(pattern)
            memory.store(pattern, f'pattern{i}')
        
        # Should only keep last 3 patterns (FIFO)
        assert memory.size() == 3
        
        # First two patterns should be evicted
        recalls = memory.recall(patterns[0], k=1)
        assert len(recalls) == 0 or recalls[0][1] < 0.9
        
        recalls = memory.recall(patterns[1], k=1)
        assert len(recalls) == 0 or recalls[0][1] < 0.9
        
        # Last three should still be there
        for i in range(2, 5):
            recalls = memory.recall(patterns[i], k=1)
            assert len(recalls) > 0
            assert recalls[0][0] == f'pattern{i}'
    
    def test_cleanup_operation(self):
        """Test cleanup operation."""
        hdc = HDComputePython(dim=100)
        memory = AssociativeMemory(hdc)
        
        # Store clean pattern
        clean_pattern = hdc.random_hv()
        memory.store(clean_pattern, 'clean')
        
        # Create noisy version
        noisy_pattern = hdc.bind(clean_pattern, hdc.random_hv())
        
        # Cleanup should return something similar to stored pattern
        cleaned = memory.cleanup(noisy_pattern)
        
        # Cleaned pattern should be more similar to original than noisy
        sim_cleaned = hdc.cosine_similarity(clean_pattern, cleaned)
        sim_noisy = hdc.cosine_similarity(clean_pattern, noisy_pattern)
        
        assert sim_cleaned >= sim_noisy
    
    def test_clear_memory(self):
        """Test clearing memory."""
        hdc = HDComputePython(dim=100)
        memory = AssociativeMemory(hdc)
        
        # Store some patterns
        for i in range(3):
            pattern = hdc.random_hv()
            memory.store(pattern, f'pattern{i}')
        
        assert memory.size() == 3
        
        memory.clear()
        
        assert memory.size() == 0
        assert len(memory.patterns) == 0
        assert len(memory.labels) == 0
    
    def test_get_statistics(self):
        """Test memory statistics."""
        hdc = HDComputePython(dim=100)
        memory = AssociativeMemory(hdc, capacity=10)
        
        # Initially empty
        stats = memory.get_statistics()
        assert stats['size'] == 0
        assert stats['capacity'] == 10
        assert stats['utilization'] == 0.0
        assert stats['unique_labels'] == 0
        
        # Add some patterns
        memory.store(hdc.random_hv(), 'label1')
        memory.store(hdc.random_hv(), 'label2')
        memory.store(hdc.random_hv(), 'label1')  # Duplicate label
        
        stats = memory.get_statistics()
        assert stats['size'] == 3
        assert stats['utilization'] == 0.3
        assert stats['unique_labels'] == 2


class TestMemoryIntegration:
    """Test integration between different memory components."""
    
    def test_item_memory_with_associative_memory(self):
        """Test using ItemMemory with AssociativeMemory."""
        hdc = HDComputePython(dim=100)
        
        # Create item memory with categories
        categories = ['animal', 'fruit', 'color']
        item_memory = ItemMemory(hdc, categories)
        
        # Create associative memory for category associations
        assoc_memory = AssociativeMemory(hdc)
        
        # Store category patterns
        for category in categories:
            cat_hv = item_memory.get_hv(category)
            assoc_memory.store(cat_hv, f'category_{category}')
        
        # Query for a category
        animal_hv = item_memory.get_hv('animal')
        recalls = assoc_memory.recall(animal_hv, k=1)
        
        assert len(recalls) == 1
        assert recalls[0][0] == 'category_animal'
        assert recalls[0][1] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])