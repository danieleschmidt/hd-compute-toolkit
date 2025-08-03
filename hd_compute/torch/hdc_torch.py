"""PyTorch implementation of hyperdimensional computing operations."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple
import numpy as np

from ..core.hdc import HDCompute


class HDComputeTorch(HDCompute):
    """PyTorch-based hyperdimensional computing implementation."""
    
    def __init__(self, dim: int, device: Optional[str] = None, dtype: torch.dtype = torch.float32):
        """Initialize HDC context with PyTorch backend.
        
        Args:
            dim: Dimensionality of hypervectors
            device: PyTorch device ('cpu', 'cuda', etc.)
            dtype: Data type for hypervectors
        """
        super().__init__(dim, device)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(42)
    
    def random_hv(self, sparsity: float = 0.5, batch_size: Optional[int] = None) -> torch.Tensor:
        """Generate random binary hypervector(s).
        
        Args:
            sparsity: Fraction of 1s in the hypervector
            batch_size: Number of hypervectors to generate
            
        Returns:
            Binary hypervector(s) as torch.Tensor
        """
        shape = (batch_size, self.dim) if batch_size else (self.dim,)
        random_vals = torch.rand(shape, device=self.device, generator=self._generator)
        return (random_vals < sparsity).to(self.dtype)
    
    def bundle(self, hvs: List[torch.Tensor], threshold: Optional[float] = None) -> torch.Tensor:
        """Bundle (superposition) hypervectors using majority voting.
        
        Args:
            hvs: List of hypervectors to bundle
            threshold: Threshold for binarization (default: len(hvs)/2)
            
        Returns:
            Bundled hypervector
        """
        if not hvs:
            raise ValueError("Cannot bundle empty list of hypervectors")
        
        stacked = torch.stack(hvs, dim=0)
        summed = torch.sum(stacked, dim=0)
        
        if threshold is None:
            threshold = len(hvs) / 2
            
        return (summed > threshold).to(self.dtype)
    
    def bind(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """Bind (XOR) two hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Bound hypervector
        """
        return torch.logical_xor(hv1.bool(), hv2.bool()).to(self.dtype)
    
    def cosine_similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """Compute cosine similarity between hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Cosine similarity value
        """
        return F.cosine_similarity(hv1.float(), hv2.float(), dim=-1).item()
    
    def hamming_distance(self, hv1: torch.Tensor, hv2: torch.Tensor) -> int:
        """Compute Hamming distance between binary hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Hamming distance
        """
        return torch.sum(torch.logical_xor(hv1.bool(), hv2.bool())).item()
    
    def permute(self, hv: torch.Tensor, positions: int) -> torch.Tensor:
        """Permute hypervector by shifting positions.
        
        Args:
            hv: Input hypervector
            positions: Number of positions to shift
            
        Returns:
            Permuted hypervector
        """
        return torch.roll(hv, shifts=positions, dims=-1)
    
    def batch_cosine_similarity(self, hvs1: torch.Tensor, hvs2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarities between batches of hypervectors.
        
        Args:
            hvs1: First batch of hypervectors [batch_size, dim]
            hvs2: Second batch of hypervectors [batch_size, dim]
            
        Returns:
            Cosine similarities [batch_size]
        """
        return F.cosine_similarity(hvs1.float(), hvs2.float(), dim=-1)
    
    def cleanup(self, hv: torch.Tensor, item_memory: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Clean up noisy hypervector using item memory.
        
        Args:
            hv: Noisy hypervector to clean up
            item_memory: Memory containing clean hypervectors [num_items, dim]
            k: Number of nearest neighbors to consider
            
        Returns:
            Cleaned hypervector
        """
        similarities = F.cosine_similarity(hv.float().unsqueeze(0), item_memory.float(), dim=-1)
        _, top_indices = torch.topk(similarities, k)
        
        if k == 1:
            return item_memory[top_indices[0]]
        else:
            return self.bundle([item_memory[idx] for idx in top_indices])
    
    def encode_sequence(self, sequence: List[torch.Tensor], position_hvs: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    
    def create_item_memory(self, items: List[str], num_items: Optional[int] = None) -> Tuple[torch.Tensor, dict]:
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