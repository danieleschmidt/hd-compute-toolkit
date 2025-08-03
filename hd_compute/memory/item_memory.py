"""Item memory for encoding discrete symbols as hypervectors."""

from typing import Dict, List, Optional, Union, Any
import numpy as np


class ItemMemory:
    """Memory structure for storing and retrieving item hypervectors."""
    
    def __init__(self, hdc_backend: Any, items: Optional[List[str]] = None):
        """Initialize item memory.
        
        Args:
            hdc_backend: HDCompute backend (PyTorch or JAX)
            items: Initial list of items to store
        """
        self.hdc = hdc_backend
        self.items: List[str] = []
        self.item_to_index: Dict[str, int] = {}
        self.memory = None
        
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
        
        new_hvs = self.hdc.random_hv(batch_size=len(new_items))
        
        if self.memory is None:
            self.memory = new_hvs
        else:
            if hasattr(self.memory, 'cat'):  # PyTorch
                import torch
                self.memory = torch.cat([self.memory, new_hvs], dim=0)
            else:  # JAX
                import jax.numpy as jnp
                self.memory = jnp.concatenate([self.memory, new_hvs], axis=0)
    
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
    
    def get_multiple_hvs(self, items: List[str]) -> Any:
        """Get hypervectors for multiple items.
        
        Args:
            items: List of item names
            
        Returns:
            Batch of hypervectors [len(items), dim]
        """
        indices = [self.item_to_index[item] for item in items]
        return self.memory[indices]
    
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
        if self.memory is None:
            raise ValueError("Memory is empty")
        
        if hasattr(self.memory, 'device'):  # PyTorch
            similarities = self.hdc.batch_cosine_similarity(
                noisy_hv.unsqueeze(0).expand(self.memory.shape[0], -1),
                self.memory
            )
            best_idx = similarities.argmax().item()
        else:  # JAX
            import jax.numpy as jnp
            from jax import vmap
            similarities = vmap(lambda mem_hv: self.hdc.cosine_similarity(noisy_hv, mem_hv))(self.memory)
            best_idx = jnp.argmax(similarities).item()
        
        return self.items[best_idx]
    
    def size(self) -> int:
        """Get number of items in memory."""
        return len(self.items)
    
    def clear(self) -> None:
        """Clear all items from memory."""
        self.items.clear()
        self.item_to_index.clear()
        self.memory = None