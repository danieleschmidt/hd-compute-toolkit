"""Simple memory implementations that work with any backend."""

from typing import Dict, List, Optional, Union, Any


class SimpleItemMemory:
    """Simple item memory for storing and retrieving item hypervectors."""
    
    def __init__(self, hdc_backend: Any, items: Optional[List[str]] = None):
        """Initialize item memory.
        
        Args:
            hdc_backend: HDCompute backend instance
            items: Initial list of items to store
        """
        self.hdc = hdc_backend
        self.items: List[str] = []
        self.item_to_index: Dict[str, int] = {}
        self.memory = []  # List of hypervectors
        
        if items:
            self.add_items(items)
    
    def add_items(self, items: List[str]) -> None:
        """Add new items to memory.
        
        Args:
            items: List of item names to add
        """
        new_items = [item for item in items if item not in self.item_to_index]
        if not new_items:
            return
        
        start_idx = len(self.items)
        self.items.extend(new_items)
        
        for i, item in enumerate(new_items):
            self.item_to_index[item] = start_idx + i
        
        # Generate new hypervectors for new items
        for _ in new_items:
            new_hv = self.hdc.random_hv()
            self.memory.append(new_hv)
    
    def get_hv(self, item: str) -> Any:
        """Get hypervector for an item.
        
        Args:
            item: Item name
            
        Returns:
            Hypervector representing the item
        """
        if item not in self.item_to_index:
            raise KeyError(f"Item '{item}' not found in memory")
        
        idx = self.item_to_index[item]
        return self.memory[idx]
    
    def get_multiple_hvs(self, items: List[str]) -> List[Any]:
        """Get hypervectors for multiple items.
        
        Args:
            items: List of item names
            
        Returns:
            List of hypervectors
        """
        return [self.get_hv(item) for item in items]
    
    def encode_sequence(self, sequence: List[str]) -> Any:
        """Encode a sequence of items with positional information.
        
        Args:
            sequence: List of item names
            
        Returns:
            Hypervector encoding the sequence
        """
        if not all(item in self.item_to_index for item in sequence):
            missing = [item for item in sequence if item not in self.item_to_index]
            raise KeyError(f"Items not found in memory: {missing}")
        
        item_hvs = [self.get_hv(item) for item in sequence]
        return self.hdc.encode_sequence(item_hvs)
    
    def cleanup(self, noisy_hv: Any, k: int = 1) -> str:
        """Clean up noisy hypervector and return best matching item.
        
        Args:
            noisy_hv: Noisy hypervector to clean up
            k: Number of nearest neighbors to consider
            
        Returns:
            Best matching item name
        """
        if not self.memory:
            raise ValueError("Memory is empty")
        
        # Compute similarities to all items
        similarities = []
        for i, mem_hv in enumerate(self.memory):
            similarity = self.hdc.cosine_similarity(noisy_hv, mem_hv)
            similarities.append((i, similarity))
        
        # Find best match
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_idx = similarities[0][0]
        
        return self.items[best_idx]
    
    def size(self) -> int:
        """Get number of items in memory."""
        return len(self.items)
    
    def clear(self) -> None:
        """Clear all items from memory."""
        self.items.clear()
        self.item_to_index.clear()
        self.memory.clear()


class SimpleAssociativeMemory:
    """Simple associative memory for pattern storage and retrieval."""
    
    def __init__(self, hdc_backend: Any, capacity: int = 1000):
        """Initialize associative memory.
        
        Args:
            hdc_backend: HDCompute backend instance
            capacity: Maximum number of patterns to store
        """
        self.hdc = hdc_backend
        self.capacity = capacity
        self.patterns: List[Any] = []
        self.labels: List[str] = []
    
    def store(self, pattern: Any, label: str) -> None:
        """Store a pattern-label association.
        
        Args:
            pattern: Input hypervector pattern
            label: Associated label
        """
        if len(self.patterns) >= self.capacity:
            # Remove oldest pattern (FIFO)
            self.patterns.pop(0)
            self.labels.pop(0)
        
        self.patterns.append(pattern)
        self.labels.append(label)
    
    def recall(self, query: Any, k: int = 1, threshold: float = 0.0) -> List[tuple]:
        """Recall patterns similar to query.
        
        Args:
            query: Query hypervector
            k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (label, similarity) tuples
        """
        if not self.patterns:
            return []
        
        # Compute similarities
        similarities = []
        for i, pattern in enumerate(self.patterns):
            similarity = self.hdc.cosine_similarity(query, pattern)
            if similarity >= threshold:
                similarities.append((self.labels[i], similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def cleanup(self, noisy_pattern: Any) -> Any:
        """Clean up noisy pattern using stored patterns.
        
        Args:
            noisy_pattern: Noisy input pattern
            
        Returns:
            Cleaned pattern (closest stored pattern)
        """
        if not self.patterns:
            return noisy_pattern
        
        recalls = self.recall(noisy_pattern, k=1)
        if not recalls:
            return noisy_pattern
        
        # Find the index of the best match
        best_label = recalls[0][0]
        best_idx = self.labels.index(best_label)
        return self.patterns[best_idx]
    
    def size(self) -> int:
        """Get number of patterns stored."""
        return len(self.patterns)
    
    def clear(self) -> None:
        """Clear all stored patterns."""
        self.patterns.clear()
        self.labels.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.patterns:
            return {"size": 0, "capacity": self.capacity, "utilization": 0.0}
        
        return {
            "size": len(self.patterns),
            "capacity": self.capacity,
            "utilization": len(self.patterns) / self.capacity,
            "unique_labels": len(set(self.labels))
        }