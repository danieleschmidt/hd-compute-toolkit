"""JAX implementation of hyperdimensional computing operations."""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from typing import List, Optional, Union, Tuple, Any
import numpy as np

from ..core.hdc import HDCompute


class HDComputeJAX(HDCompute):
    """JAX-based hyperdimensional computing implementation."""
    
    def __init__(self, dim: int, key: Optional[jax.random.PRNGKey] = None, device: Optional[str] = None):
        """Initialize HDC context with JAX backend.
        
        Args:
            dim: Dimensionality of hypervectors
            key: JAX random key for reproducibility
            device: JAX device (auto-detected if None)
        """
        super().__init__(dim, device)
        self.key = key if key is not None else random.PRNGKey(42)
        self._key_counter = 0
    
    def _next_key(self) -> jax.random.PRNGKey:
        """Get next random key."""
        self.key, subkey = random.split(self.key)
        return subkey
    
    def random_hv(self, sparsity: float = 0.5, batch_size: Optional[int] = None) -> jnp.ndarray:
        """Generate random binary hypervector(s).
        
        Args:
            sparsity: Fraction of 1s in the hypervector
            batch_size: Number of hypervectors to generate
            
        Returns:
            Binary hypervector(s) as jnp.ndarray
        """
        shape = (batch_size, self.dim) if batch_size else (self.dim,)
        random_vals = random.uniform(self._next_key(), shape)
        return (random_vals < sparsity).astype(jnp.float32)
    
    @jit
    def bundle(self, hvs: jnp.ndarray, threshold: Optional[float] = None) -> jnp.ndarray:
        """Bundle (superposition) hypervectors using majority voting.
        
        Args:
            hvs: Array of hypervectors to bundle [num_hvs, dim]
            threshold: Threshold for binarization
            
        Returns:
            Bundled hypervector
        """
        summed = jnp.sum(hvs, axis=0)
        if threshold is None:
            threshold = hvs.shape[0] / 2
        return (summed > threshold).astype(jnp.float32)
    
    @jit
    def bind(self, hv1: jnp.ndarray, hv2: jnp.ndarray) -> jnp.ndarray:
        """Bind (XOR) two hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Bound hypervector
        """
        return jnp.logical_xor(hv1.astype(bool), hv2.astype(bool)).astype(jnp.float32)
    
    @jit
    def cosine_similarity(self, hv1: jnp.ndarray, hv2: jnp.ndarray) -> float:
        """Compute cosine similarity between hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Cosine similarity value
        """
        dot_product = jnp.dot(hv1, hv2)
        norm1 = jnp.linalg.norm(hv1)
        norm2 = jnp.linalg.norm(hv2)
        return dot_product / (norm1 * norm2 + 1e-8)
    
    @jit
    def hamming_distance(self, hv1: jnp.ndarray, hv2: jnp.ndarray) -> int:
        """Compute Hamming distance between binary hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Hamming distance
        """
        return jnp.sum(jnp.logical_xor(hv1.astype(bool), hv2.astype(bool)))
    
    @jit
    def permute(self, hv: jnp.ndarray, positions: int) -> jnp.ndarray:
        """Permute hypervector by shifting positions.
        
        Args:
            hv: Input hypervector
            positions: Number of positions to shift
            
        Returns:
            Permuted hypervector
        """
        return jnp.roll(hv, shift=positions, axis=-1)
    
    @jit
    def batch_cosine_similarity(self, hvs1: jnp.ndarray, hvs2: jnp.ndarray) -> jnp.ndarray:
        """Compute cosine similarities between batches of hypervectors.
        
        Args:
            hvs1: First batch of hypervectors [batch_size, dim]
            hvs2: Second batch of hypervectors [batch_size, dim]
            
        Returns:
            Cosine similarities [batch_size]
        """
        return vmap(self.cosine_similarity)(hvs1, hvs2)
    
    def cleanup(self, hv: jnp.ndarray, item_memory: jnp.ndarray, k: int = 1) -> jnp.ndarray:
        """Clean up noisy hypervector using item memory.
        
        Args:
            hv: Noisy hypervector to clean up
            item_memory: Memory containing clean hypervectors [num_items, dim]
            k: Number of nearest neighbors to consider
            
        Returns:
            Cleaned hypervector
        """
        similarities = vmap(lambda mem_hv: self.cosine_similarity(hv, mem_hv))(item_memory)
        top_indices = jnp.argsort(similarities)[-k:]
        
        if k == 1:
            return item_memory[top_indices[-1]]
        else:
            selected_hvs = item_memory[top_indices]
            return self.bundle(selected_hvs)
    
    def encode_sequence(self, sequence: List[jnp.ndarray], position_hvs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Encode a sequence of hypervectors with positional information.
        
        Args:
            sequence: List of hypervectors representing sequence elements
            position_hvs: Position hypervectors [seq_len, dim]
            
        Returns:
            Encoded sequence hypervector
        """
        if position_hvs is None:
            position_hvs = self.random_hv(batch_size=len(sequence))
        
        bound_elements = []
        for i, element in enumerate(sequence):
            bound = self.bind(element, position_hvs[i])
            bound_elements.append(bound)
        
        bound_array = jnp.stack(bound_elements)
        return self.bundle(bound_array)
    
    def create_item_memory(self, items: List[str], num_items: Optional[int] = None) -> Tuple[jnp.ndarray, dict]:
        """Create item memory for symbol encoding.
        
        Args:
            items: List of item names/symbols
            num_items: Number of items (default: len(items))
            
        Returns:
            Tuple of (memory tensor [num_items, dim], item_to_index mapping)
        """
        if num_items is None:
            num_items = len(items)
        
        memory = self.random_hv(batch_size=num_items)
        item_to_index = {item: i for i, item in enumerate(items)}
        
        return memory, item_to_index