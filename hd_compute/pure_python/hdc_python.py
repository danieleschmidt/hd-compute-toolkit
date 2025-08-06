"""Pure Python implementation of hyperdimensional computing operations."""

import random
import math
from typing import List, Optional, Union, Tuple
import warnings
import logging

from ..core.hdc import HDCompute

logger = logging.getLogger(__name__)

# Try to import validation utilities
try:
    from ..utils.validation import (
        ParameterValidator, validate_hypervector, validate_hypervector_list,
        safe_operation, HDCValidationError, DimensionMismatchError
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    logger.debug("Validation utilities not available, running without validation")


class SimpleArray:
    """Simple array-like class using Python lists."""
    
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self.data = list(data)
            self.shape = shape or (len(data),)
        elif isinstance(data, (int, float)):
            # Single value
            self.data = [float(data)]
            self.shape = (1,)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value
    
    def tolist(self):
        return self.data.copy()
    
    def astype(self, dtype):
        if dtype == bool:
            return SimpleArray([bool(x) for x in self.data], self.shape)
        elif dtype == float:
            return SimpleArray([float(x) for x in self.data], self.shape)
        elif dtype == int:
            return SimpleArray([int(x) for x in self.data], self.shape)
        else:
            return self
    
    def sum(self):
        return sum(self.data)
    
    def dot(self, other):
        """Compute dot product with another array."""
        if len(self.data) != len(other.data):
            raise ValueError("Arrays must have same length")
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def norm(self):
        """Compute L2 norm."""
        return math.sqrt(sum(x * x for x in self.data))


class HDComputePython(HDCompute):
    """Pure Python hyperdimensional computing implementation."""
    
    def __init__(self, dim: int, device: Optional[str] = None, dtype=float):
        """Initialize HDC context with Pure Python backend.
        
        Args:
            dim: Dimensionality of hypervectors
            device: Device specification (ignored for Python)
            dtype: Data type for hypervectors
        """
        # Validate parameters if validation is available
        if VALIDATION_AVAILABLE:
            validated_dim, validated_device = ParameterValidator.validate_hdc_init_params(dim, device)
        else:
            validated_dim, validated_device = dim, device
            
        super().__init__(validated_dim, validated_device)
        self.dtype = dtype
        self._rng = random.Random(42)
        
        if device and device != 'cpu':
            warnings.warn(f"Device '{device}' not supported in Python backend, using CPU")
    
    def random_hv(self, sparsity: float = 0.5, batch_size: Optional[int] = None) -> SimpleArray:
        """Generate random binary hypervector(s).
        
        Args:
            sparsity: Fraction of 1s in the hypervector
            batch_size: Number of hypervectors to generate (not supported in single mode)
            
        Returns:
            Binary hypervector as SimpleArray
        """
        # Validate parameters if validation is available
        if VALIDATION_AVAILABLE:
            try:
                validated_sparsity, validated_batch_size = ParameterValidator.validate_random_hv_params(sparsity, batch_size)
            except Exception as e:
                logger.error(f"Parameter validation failed: {e}")
                raise
        else:
            validated_sparsity, validated_batch_size = sparsity, batch_size
        
        if validated_batch_size:
            # Return first hypervector only for simplicity
            warnings.warn("Batch generation not fully supported in Python backend")
        
        try:
            data = []
            for _ in range(self.dim):
                value = 1.0 if self._rng.random() < validated_sparsity else 0.0
                data.append(value)
            
            return SimpleArray(data, (self.dim,))
        except Exception as e:
            logger.error(f"Failed to generate random hypervector: {e}")
            raise
    
    def bundle(self, hvs: List[SimpleArray], threshold: Optional[float] = None) -> SimpleArray:
        """Bundle (superposition) hypervectors using majority voting.
        
        Args:
            hvs: List of hypervectors to bundle
            threshold: Threshold for binarization (default: len(hvs)/2)
            
        Returns:
            Bundled hypervector
        """
        # Validate inputs if validation is available
        if VALIDATION_AVAILABLE:
            try:
                validated_hvs = validate_hypervector_list(hvs, self.dim)
            except Exception as e:
                logger.error(f"Hypervector validation failed: {e}")
                raise
        else:
            validated_hvs = hvs
            if not hvs:
                raise ValueError("Cannot bundle empty list of hypervectors")
        
        if threshold is None:
            threshold = len(validated_hvs) / 2
        
        try:
            # Sum all hypervectors element-wise
            bundled_data = [0.0] * self.dim
            for hv in validated_hvs:
                for i in range(min(len(hv.data), self.dim)):
                    bundled_data[i] += hv.data[i]
            
            # Apply threshold
            result_data = [1.0 if val > threshold else 0.0 for val in bundled_data]
            return SimpleArray(result_data, (self.dim,))
        except Exception as e:
            logger.error(f"Failed to bundle hypervectors: {e}")
            raise
    
    def bind(self, hv1: SimpleArray, hv2: SimpleArray) -> SimpleArray:
        """Bind (XOR) two hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Bound hypervector
        """
        if len(hv1.data) != len(hv2.data):
            raise ValueError("Hypervectors must have same length")
        
        # XOR operation: result is 1 if inputs differ, 0 if same
        result_data = []
        for a, b in zip(hv1.data, hv2.data):
            # Convert to boolean, XOR, convert back to float
            bool_a = bool(a > 0.5)
            bool_b = bool(b > 0.5)
            result_data.append(1.0 if bool_a != bool_b else 0.0)
        
        return SimpleArray(result_data, (self.dim,))
    
    def cosine_similarity(self, hv1: SimpleArray, hv2: SimpleArray) -> float:
        """Compute cosine similarity between hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Cosine similarity value
        """
        dot_product = hv1.dot(hv2)
        norm1 = hv1.norm()
        norm2 = hv2.norm()
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def hamming_distance(self, hv1: SimpleArray, hv2: SimpleArray) -> int:
        """Compute Hamming distance between binary hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Hamming distance
        """
        if len(hv1.data) != len(hv2.data):
            raise ValueError("Hypervectors must have same length")
        
        distance = 0
        for a, b in zip(hv1.data, hv2.data):
            bool_a = bool(a > 0.5)
            bool_b = bool(b > 0.5)
            if bool_a != bool_b:
                distance += 1
        
        return distance
    
    def permute(self, hv: SimpleArray, positions: int) -> SimpleArray:
        """Permute hypervector by shifting positions.
        
        Args:
            hv: Input hypervector
            positions: Number of positions to shift
            
        Returns:
            Permuted hypervector
        """
        n = len(hv.data)
        positions = positions % n  # Handle large shifts
        
        # Circular shift: move elements from end to beginning
        shifted_data = hv.data[-positions:] + hv.data[:-positions]
        return SimpleArray(shifted_data, hv.shape)
    
    def batch_cosine_similarity(self, hvs1: List[SimpleArray], hvs2: List[SimpleArray]) -> List[float]:
        """Compute cosine similarities between batches of hypervectors.
        
        Args:
            hvs1: First batch of hypervectors
            hvs2: Second batch of hypervectors
            
        Returns:
            List of cosine similarities
        """
        if len(hvs1) != len(hvs2):
            raise ValueError("Batches must have same size")
        
        return [self.cosine_similarity(hv1, hv2) for hv1, hv2 in zip(hvs1, hvs2)]
    
    def cleanup(self, hv: SimpleArray, item_memory: List[SimpleArray], k: int = 1) -> SimpleArray:
        """Clean up noisy hypervector using item memory.
        
        Args:
            hv: Noisy hypervector to clean up
            item_memory: Memory containing clean hypervectors
            k: Number of nearest neighbors to consider
            
        Returns:
            Cleaned hypervector
        """
        if not item_memory:
            return hv
        
        # Compute similarities to all items in memory
        similarities = [(i, self.cosine_similarity(hv, mem_hv)) 
                       for i, mem_hv in enumerate(item_memory)]
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        if k == 1:
            return item_memory[top_k[0][0]]
        else:
            # Bundle top k items
            top_hvs = [item_memory[idx] for idx, _ in top_k]
            return self.bundle(top_hvs)
    
    def encode_sequence(self, sequence: List[SimpleArray], position_hvs: Optional[List[SimpleArray]] = None) -> SimpleArray:
        """Encode a sequence of hypervectors with positional information.
        
        Args:
            sequence: List of hypervectors representing sequence elements
            position_hvs: Position hypervectors
            
        Returns:
            Encoded sequence hypervector
        """
        if not sequence:
            return self.random_hv()
        
        if position_hvs is None:
            position_hvs = [self.random_hv() for _ in sequence]
        
        bound_elements = []
        for element, pos_hv in zip(sequence, position_hvs):
            bound = self.bind(element, pos_hv)
            bound_elements.append(bound)
        
        return self.bundle(bound_elements)
    
    def create_item_memory(self, items: List[str], num_items: Optional[int] = None) -> Tuple[List[SimpleArray], dict]:
        """Create item memory for symbol encoding.
        
        Args:
            items: List of item names/symbols
            num_items: Number of items (default: len(items))
            
        Returns:
            Tuple of (memory list, item_to_index mapping)
        """
        if num_items is None:
            num_items = len(items)
        
        memory = [self.random_hv() for _ in range(num_items)]
        item_to_index = {item: i for i, item in enumerate(items)}
        
        return memory, item_to_index