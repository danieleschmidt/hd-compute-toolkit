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
    
    # Advanced similarity metrics
    
    def jensen_shannon_divergence(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """Jensen-Shannon divergence for probabilistic hypervectors."""
        # Convert to probability distributions
        p1 = torch.abs(hv1.float()) / (torch.sum(torch.abs(hv1.float())) + 1e-8)
        p2 = torch.abs(hv2.float()) / (torch.sum(torch.abs(hv2.float())) + 1e-8)
        
        # Compute average distribution
        m = 0.5 * (p1 + p2)
        
        # Compute KL divergences
        kl_p1m = torch.sum(p1 * torch.log((p1 + 1e-8) / (m + 1e-8)))
        kl_p2m = torch.sum(p2 * torch.log((p2 + 1e-8) / (m + 1e-8)))
        
        # Jensen-Shannon divergence
        return (0.5 * (kl_p1m + kl_p2m)).item()
    
    def wasserstein_distance(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """Wasserstein distance for geometric hypervector comparison."""
        # Simplified 1D Wasserstein distance using sorted values
        sorted_hv1, _ = torch.sort(hv1.flatten())
        sorted_hv2, _ = torch.sort(hv2.flatten())
        return torch.mean(torch.abs(sorted_hv1 - sorted_hv2)).item()
    
    # Novel research operations
    
    def fractional_bind(self, hv1: torch.Tensor, hv2: torch.Tensor, power: float = 0.5) -> torch.Tensor:
        """Fractional binding operation for gradual associations."""
        # Convert to float for fractional operations
        hv1_float = hv1.float()
        hv2_float = hv2.float()
        
        # Linear interpolation based binding
        bound = power * torch.logical_xor(hv1.bool(), hv2.bool()).float()
        unbound = (1 - power) * hv1_float
        
        result = bound + unbound
        # Binarize with adaptive threshold
        threshold = torch.mean(result)
        return (result > threshold).to(self.dtype)
    
    def quantum_superposition(self, hvs: List[torch.Tensor], amplitudes: Optional[List[float]] = None) -> torch.Tensor:
        """Quantum-inspired superposition with probability amplitudes."""
        if not hvs:
            raise ValueError("Cannot create superposition from empty list")
        
        if amplitudes is None:
            amplitudes = [1.0 / len(hvs)] * len(hvs)
        
        # Normalize amplitudes
        amplitudes = torch.tensor(amplitudes, device=self.device)
        amplitudes = amplitudes / torch.sum(amplitudes)
        
        # Weighted superposition
        result = torch.zeros(self.dim, device=self.device, dtype=torch.float32)
        for hv, amp in zip(hvs, amplitudes):
            result += amp * hv.float()
        
        # Probabilistic binarization based on amplitudes
        probabilities = torch.abs(result)
        probabilities = probabilities / (torch.max(probabilities) + 1e-8)
        random_vals = torch.rand_like(probabilities, generator=self._generator)
        
        return (random_vals < probabilities).to(self.dtype)
    
    def entanglement_measure(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """Measure quantum-like entanglement between hypervectors."""
        # Convert to probability distributions
        p1 = torch.abs(hv1.float()) / (torch.sum(torch.abs(hv1.float())) + 1e-8)
        p2 = torch.abs(hv2.float()) / (torch.sum(torch.abs(hv2.float())) + 1e-8)
        
        # Simplified mutual information calculation
        # Use correlation as proxy for entanglement
        correlation = torch.corrcoef(torch.stack([p1, p2]))[0, 1]
        
        # Normalize to [0,1] and handle NaN
        entanglement = torch.abs(correlation) if not torch.isnan(correlation) else 0.0
        return float(torch.clamp(entanglement, 0, 1).item())
    
    def coherence_decay(self, hv: torch.Tensor, decay_rate: float = 0.1) -> torch.Tensor:
        """Apply coherence decay to simulate memory degradation."""
        # Add noise proportional to decay rate
        noise = torch.rand_like(hv.float(), generator=self._generator) * decay_rate
        decayed = hv.float() + noise
        
        # Maintain binary nature with probability-based thresholding
        probabilities = torch.abs(decayed)
        probabilities = probabilities / (torch.max(probabilities) + 1e-8)
        random_vals = torch.rand_like(probabilities, generator=self._generator)
        
        return (random_vals < probabilities * (1 - decay_rate)).to(self.dtype)
    
    def adaptive_threshold(self, hv: torch.Tensor, target_sparsity: float = 0.5) -> torch.Tensor:
        """Adaptive thresholding to maintain target sparsity."""
        hv_float = hv.float()
        
        # Find threshold that gives target sparsity
        sorted_vals, _ = torch.sort(hv_float)
        threshold_idx = int((1 - target_sparsity) * len(sorted_vals))
        threshold = sorted_vals[threshold_idx] if threshold_idx < len(sorted_vals) else sorted_vals[-1]
        
        return (hv_float >= threshold).to(self.dtype)
    
    # Hierarchical and compositional operations
    
    def hierarchical_bind(self, structure: dict) -> torch.Tensor:
        """Hierarchical binding for complex compositional structures."""
        def _bind_structure(struct):
            if isinstance(struct, torch.Tensor):
                return struct
            elif isinstance(struct, dict):
                if not struct:
                    return self.random_hv()
                
                bound_items = []
                for key, value in struct.items():
                    key_hv = self.random_hv()  # Should use item memory in practice
                    value_hv = _bind_structure(value)
                    bound_pair = self.bind(key_hv, value_hv)
                    bound_items.append(bound_pair)
                
                return self.bundle(bound_items)
            elif isinstance(struct, list):
                if not struct:
                    return self.random_hv()
                
                bound_items = [_bind_structure(item) for item in struct]
                return self.bundle(bound_items)
            else:
                # Convert scalar to hypervector
                return self.random_hv()
        
        return _bind_structure(structure)
    
    def semantic_projection(self, hv: torch.Tensor, basis_hvs: List[torch.Tensor]) -> List[float]:
        """Project hypervector onto semantic basis."""
        if not basis_hvs:
            return []
        
        # Compute similarities with each basis vector
        similarities = []
        for basis_hv in basis_hvs:
            sim = self.cosine_similarity(hv, basis_hv)
            similarities.append(sim)
        
        # Normalize to get projection coefficients
        similarities = torch.tensor(similarities, device=self.device)
        norm = torch.norm(similarities)
        
        if norm > 1e-8:
            return (similarities / norm).tolist()
        else:
            return [0.0] * len(basis_hvs)