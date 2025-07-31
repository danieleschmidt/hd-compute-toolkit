"""Base HDCompute class with framework-agnostic interface."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union


class HDCompute(ABC):
    """Abstract base class for hyperdimensional computing operations."""
    
    def __init__(self, dim: int, device: Optional[str] = None):
        """Initialize HDC context.
        
        Args:
            dim: Dimensionality of hypervectors
            device: Computing device ('cpu', 'cuda', 'tpu', etc.)
        """
        self.dim = dim
        self.device = device
    
    @abstractmethod
    def random_hv(self, sparsity: float = 0.5) -> Any:
        """Generate random hypervector."""
        pass
    
    @abstractmethod
    def bundle(self, hvs: List[Any]) -> Any:
        """Bundle (superposition) of hypervectors."""
        pass
    
    @abstractmethod  
    def bind(self, hv1: Any, hv2: Any) -> Any:
        """Bind (association) two hypervectors."""
        pass
    
    @abstractmethod
    def cosine_similarity(self, hv1: Any, hv2: Any) -> float:
        """Compute cosine similarity between hypervectors."""
        pass