"""NumPy implementation of hyperdimensional computing operations."""

import numpy as np
from typing import List, Optional, Union, Tuple
import warnings

from ..core.hdc import HDCompute


class HDComputeNumPy(HDCompute):
    """NumPy-based hyperdimensional computing implementation."""
    
    def __init__(self, dim: int, device: Optional[str] = None, dtype: np.dtype = np.float32):
        """Initialize HDC context with NumPy backend.
        
        Args:
            dim: Dimensionality of hypervectors
            device: Device specification (ignored for NumPy)
            dtype: Data type for hypervectors
        """
        super().__init__(dim, device)
        self.dtype = dtype
        self._rng = np.random.RandomState(42)
        
        if device and device != 'cpu':
            warnings.warn(f"Device '{device}' not supported in NumPy backend, using CPU")
    
    def random_hv(self, sparsity: float = 0.5, batch_size: Optional[int] = None) -> np.ndarray:
        """Generate random binary hypervector(s).
        
        Args:
            sparsity: Fraction of 1s in the hypervector
            batch_size: Number of hypervectors to generate
            
        Returns:
            Binary hypervector(s) as np.ndarray
        """
        shape = (batch_size, self.dim) if batch_size else (self.dim,)
        random_vals = self._rng.rand(*shape)
        return (random_vals < sparsity).astype(self.dtype)
    
    def bundle(self, hvs: List[np.ndarray], threshold: Optional[float] = None) -> np.ndarray:
        """Bundle (superposition) hypervectors using majority voting.
        
        Args:
            hvs: List of hypervectors to bundle
            threshold: Threshold for binarization (default: len(hvs)/2)
            
        Returns:
            Bundled hypervector
        """
        if not hvs:
            raise ValueError("Cannot bundle empty list of hypervectors")
        
        stacked = np.stack(hvs, axis=0)
        summed = np.sum(stacked, axis=0)
        
        if threshold is None:
            threshold = len(hvs) / 2
            
        return (summed > threshold).astype(self.dtype)
    
    def bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Bind (XOR) two hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Bound hypervector
        """
        return np.logical_xor(hv1.astype(bool), hv2.astype(bool)).astype(self.dtype)
    
    def cosine_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute cosine similarity between hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Cosine similarity value
        """
        dot_product = np.dot(hv1, hv2)
        norm1 = np.linalg.norm(hv1)
        norm2 = np.linalg.norm(hv2)
        return float(dot_product / (norm1 * norm2 + 1e-8))
    
    def hamming_distance(self, hv1: np.ndarray, hv2: np.ndarray) -> int:
        """Compute Hamming distance between binary hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Hamming distance
        """
        return int(np.sum(np.logical_xor(hv1.astype(bool), hv2.astype(bool))))
    
    def permute(self, hv: np.ndarray, positions: int) -> np.ndarray:
        """Permute hypervector by shifting positions.
        
        Args:
            hv: Input hypervector
            positions: Number of positions to shift
            
        Returns:
            Permuted hypervector
        """
        return np.roll(hv, shift=positions, axis=-1)
    
    def batch_cosine_similarity(self, hvs1: np.ndarray, hvs2: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between batches of hypervectors.
        
        Args:
            hvs1: First batch of hypervectors [batch_size, dim]
            hvs2: Second batch of hypervectors [batch_size, dim]
            
        Returns:
            Cosine similarities [batch_size]
        """
        # Vectorized cosine similarity computation
        dot_products = np.sum(hvs1 * hvs2, axis=1)
        norms1 = np.linalg.norm(hvs1, axis=1)
        norms2 = np.linalg.norm(hvs2, axis=1)
        return dot_products / (norms1 * norms2 + 1e-8)
    
    def cleanup(self, hv: np.ndarray, item_memory: np.ndarray, k: int = 1) -> np.ndarray:
        """Clean up noisy hypervector using item memory.
        
        Args:
            hv: Noisy hypervector to clean up
            item_memory: Memory containing clean hypervectors [num_items, dim]
            k: Number of nearest neighbors to consider
            
        Returns:
            Cleaned hypervector
        """
        # Compute similarities to all items in memory
        hv_expanded = np.expand_dims(hv, 0).repeat(item_memory.shape[0], axis=0)
        similarities = self.batch_cosine_similarity(hv_expanded, item_memory)
        
        # Get top k matches
        top_indices = np.argsort(similarities)[-k:]
        
        if k == 1:
            return item_memory[top_indices[-1]]
        else:
            selected_hvs = [item_memory[idx] for idx in top_indices]
            return self.bundle(selected_hvs)
    
    def encode_sequence(self, sequence: List[np.ndarray], position_hvs: Optional[np.ndarray] = None) -> np.ndarray:
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
        
        return self.bundle(bound_elements)
    
    def create_item_memory(self, items: List[str], num_items: Optional[int] = None) -> Tuple[np.ndarray, dict]:
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