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
    
    # Advanced similarity metrics
    
    def jensen_shannon_divergence(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Jensen-Shannon divergence for probabilistic hypervectors."""
        # Convert to probability distributions
        p1 = np.abs(hv1) / (np.sum(np.abs(hv1)) + 1e-8)
        p2 = np.abs(hv2) / (np.sum(np.abs(hv2)) + 1e-8)
        
        # Compute average distribution
        m = 0.5 * (p1 + p2)
        
        # Compute KL divergences
        kl_p1m = np.sum(p1 * np.log((p1 + 1e-8) / (m + 1e-8)))
        kl_p2m = np.sum(p2 * np.log((p2 + 1e-8) / (m + 1e-8)))
        
        # Jensen-Shannon divergence
        return float(0.5 * (kl_p1m + kl_p2m))
    
    def wasserstein_distance(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Wasserstein distance for geometric hypervector comparison."""
        # Simplified 1D Wasserstein distance using sorted values
        sorted_hv1 = np.sort(hv1.flatten())
        sorted_hv2 = np.sort(hv2.flatten())
        return float(np.mean(np.abs(sorted_hv1 - sorted_hv2)))
    
    # Novel research operations
    
    def fractional_bind(self, hv1: np.ndarray, hv2: np.ndarray, power: float = 0.5) -> np.ndarray:
        """Fractional binding operation for gradual associations."""
        # Convert to float for fractional operations
        hv1_float = hv1.astype(np.float32)
        hv2_float = hv2.astype(np.float32)
        
        # Linear interpolation based binding
        bound = power * np.logical_xor(hv1.astype(bool), hv2.astype(bool)).astype(np.float32)
        unbound = (1 - power) * hv1_float
        
        result = bound + unbound
        # Binarize with adaptive threshold
        threshold = np.mean(result)
        return (result > threshold).astype(self.dtype)
    
    def quantum_superposition(self, hvs: List[np.ndarray], amplitudes: Optional[List[float]] = None) -> np.ndarray:
        """Quantum-inspired superposition with probability amplitudes."""
        if not hvs:
            raise ValueError("Cannot create superposition from empty list")
        
        if amplitudes is None:
            amplitudes = [1.0 / len(hvs)] * len(hvs)
        
        # Normalize amplitudes
        amplitudes = np.array(amplitudes)
        amplitudes = amplitudes / np.sum(amplitudes)
        
        # Weighted superposition
        result = np.zeros(self.dim, dtype=np.float32)
        for hv, amp in zip(hvs, amplitudes):
            result += amp * hv.astype(np.float32)
        
        # Probabilistic binarization based on amplitudes
        probabilities = np.abs(result)
        probabilities = probabilities / (np.max(probabilities) + 1e-8)
        random_vals = self._rng.rand(self.dim)
        
        return (random_vals < probabilities).astype(self.dtype)
    
    def entanglement_measure(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Measure quantum-like entanglement between hypervectors."""
        # Convert to probability distributions
        p1 = np.abs(hv1.astype(np.float32)) / (np.sum(np.abs(hv1)) + 1e-8)
        p2 = np.abs(hv2.astype(np.float32)) / (np.sum(np.abs(hv2)) + 1e-8)
        
        # Compute joint probability (outer product)
        joint_prob = np.outer(p1, p2).flatten()
        marginal_prob = np.outer(np.ones_like(p1), np.ones_like(p2)).flatten()
        marginal_prob = marginal_prob / np.sum(marginal_prob)
        
        # Mutual information as entanglement measure
        mutual_info = 0.0
        for jp, mp in zip(joint_prob, marginal_prob):
            if jp > 1e-8 and mp > 1e-8:
                mutual_info += jp * np.log(jp / mp)
        
        # Normalize to [0,1]
        return float(min(1.0, mutual_info / np.log(2)))
    
    def coherence_decay(self, hv: np.ndarray, decay_rate: float = 0.1) -> np.ndarray:
        """Apply coherence decay to simulate memory degradation."""
        # Add noise proportional to decay rate
        noise = self._rng.rand(self.dim) * decay_rate
        decayed = hv.astype(np.float32) + noise
        
        # Maintain binary nature with probability-based thresholding
        probabilities = np.abs(decayed)
        probabilities = probabilities / (np.max(probabilities) + 1e-8)
        random_vals = self._rng.rand(self.dim)
        
        return (random_vals < probabilities * (1 - decay_rate)).astype(self.dtype)
    
    def adaptive_threshold(self, hv: np.ndarray, target_sparsity: float = 0.5) -> np.ndarray:
        """Adaptive thresholding to maintain target sparsity."""
        hv_float = hv.astype(np.float32)
        
        # Find threshold that gives target sparsity
        sorted_vals = np.sort(hv_float)
        threshold_idx = int((1 - target_sparsity) * len(sorted_vals))
        threshold = sorted_vals[threshold_idx] if threshold_idx < len(sorted_vals) else sorted_vals[-1]
        
        return (hv_float >= threshold).astype(self.dtype)
    
    # Hierarchical and compositional operations
    
    def hierarchical_bind(self, structure: dict) -> np.ndarray:
        """Hierarchical binding for complex compositional structures."""
        def _bind_structure(struct):
            if isinstance(struct, np.ndarray):
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
    
    def semantic_projection(self, hv: np.ndarray, basis_hvs: List[np.ndarray]) -> List[float]:
        """Project hypervector onto semantic basis."""
        if not basis_hvs:
            return []
        
        # Compute similarities with each basis vector
        similarities = []
        for basis_hv in basis_hvs:
            sim = self.cosine_similarity(hv, basis_hv)
            similarities.append(sim)
        
        # Normalize to get projection coefficients
        similarities = np.array(similarities)
        norm = np.linalg.norm(similarities)
        
        if norm > 1e-8:
            return (similarities / norm).tolist()
        else:
            return [0.0] * len(basis_hvs)