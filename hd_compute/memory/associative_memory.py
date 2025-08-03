"""Associative memory for storing and retrieving patterns."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class AssociativeMemory:
    """Associative memory for pattern storage and retrieval."""
    
    def __init__(self, hdc_backend: Any, capacity: int = 1000):
        """Initialize associative memory.
        
        Args:
            hdc_backend: HDCompute backend (PyTorch or JAX)
            capacity: Maximum number of patterns to store
        """
        self.hdc = hdc_backend
        self.capacity = capacity
        self.patterns: List[Any] = []
        self.labels: List[str] = []
        self.memory = None
    
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
            if self.memory is not None:
                if hasattr(self.memory, 'device'):  # PyTorch
                    self.memory = self.memory[1:]
                else:  # JAX
                    self.memory = self.memory[1:]
        
        self.patterns.append(pattern)
        self.labels.append(label)
        
        if self.memory is None:
            self.memory = pattern.unsqueeze(0) if hasattr(pattern, 'unsqueeze') else pattern[None, :]
        else:
            if hasattr(self.memory, 'cat'):  # PyTorch
                import torch
                self.memory = torch.cat([self.memory, pattern.unsqueeze(0)], dim=0)
            else:  # JAX
                import jax.numpy as jnp
                self.memory = jnp.concatenate([self.memory, pattern[None, :]], axis=0)
    
    def recall(self, query: Any, k: int = 1, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Recall patterns similar to query.
        
        Args:
            query: Query hypervector
            k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (label, similarity) tuples
        """
        if self.memory is None or len(self.patterns) == 0:
            return []
        
        if hasattr(self.memory, 'device'):  # PyTorch
            similarities = self.hdc.batch_cosine_similarity(
                query.unsqueeze(0).expand(self.memory.shape[0], -1),
                self.memory
            )
            similarities = similarities.cpu().numpy()
        else:  # JAX
            import jax.numpy as jnp
            from jax import vmap
            similarities = vmap(lambda mem_pattern: self.hdc.cosine_similarity(query, mem_pattern))(self.memory)
            similarities = np.array(similarities)
        
        # Filter by threshold
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            return []
        
        # Get top k matches
        top_k = min(k, len(valid_indices))
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[-top_k:]][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.labels[idx], float(similarities[idx])))
        
        return results
    
    def cleanup(self, noisy_pattern: Any) -> Any:
        """Clean up noisy pattern using stored patterns.
        
        Args:
            noisy_pattern: Noisy input pattern
            
        Returns:
            Cleaned pattern (closest stored pattern)
        """
        if self.memory is None or len(self.patterns) == 0:
            return noisy_pattern
        
        recalls = self.recall(noisy_pattern, k=1)
        if not recalls:
            return noisy_pattern
        
        # Find the index of the best match
        best_label = recalls[0][0]
        best_idx = self.labels.index(best_label)
        return self.patterns[best_idx]
    
    def associative_recall(self, encoded_query: Any, item_memory: Any) -> List[str]:
        """Perform associative recall using item memory.
        
        Args:
            encoded_query: Encoded query hypervector
            item_memory: ItemMemory instance for decoding
            
        Returns:
            List of recalled item names
        """
        recalls = self.recall(encoded_query, k=5, threshold=0.1)
        results = []
        
        for label, similarity in recalls:
            try:
                # Try to decode using item memory cleanup
                decoded = item_memory.cleanup(encoded_query)
                results.append(decoded)
            except:
                # Fallback to label if cleanup fails
                results.append(label)
        
        return results
    
    def size(self) -> int:
        """Get number of patterns stored."""
        return len(self.patterns)
    
    def clear(self) -> None:
        """Clear all stored patterns."""
        self.patterns.clear()
        self.labels.clear()
        self.memory = None
    
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