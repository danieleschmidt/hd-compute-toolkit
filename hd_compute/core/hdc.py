"""Base HDCompute class with framework-agnostic interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional numpy import for statistical analysis
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


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
    
    # Novel research-oriented operations
    
    @abstractmethod
    def fractional_bind(self, hv1: Any, hv2: Any, power: float = 0.5) -> Any:
        """Fractional binding operation for gradual associations.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector  
            power: Binding strength (0=no binding, 1=full binding)
            
        Returns:
            Fractionally bound hypervector
        """
        pass
    
    @abstractmethod
    def quantum_superposition(self, hvs: List[Any], amplitudes: Optional[List[float]] = None) -> Any:
        """Quantum-inspired superposition with probability amplitudes.
        
        Args:
            hvs: List of hypervectors
            amplitudes: Probability amplitudes (normalized if None)
            
        Returns:
            Quantum superposition hypervector
        """
        pass
    
    @abstractmethod
    def entanglement_measure(self, hv1: Any, hv2: Any) -> float:
        """Measure quantum-like entanglement between hypervectors.
        
        Returns:
            Entanglement coefficient [0,1]
        """
        pass
    
    @abstractmethod
    def coherence_decay(self, hv: Any, decay_rate: float = 0.1) -> Any:
        """Apply coherence decay to simulate memory degradation.
        
        Args:
            hv: Hypervector
            decay_rate: Rate of coherence loss
            
        Returns:
            Decayed hypervector
        """
        pass
    
    @abstractmethod
    def adaptive_threshold(self, hv: Any, target_sparsity: float = 0.5) -> Any:
        """Adaptive thresholding to maintain target sparsity.
        
        Args:
            hv: Input hypervector
            target_sparsity: Desired sparsity level
            
        Returns:
            Thresholded hypervector
        """
        pass
    
    # Advanced similarity metrics
    
    @abstractmethod
    def hamming_distance(self, hv1: Any, hv2: Any) -> float:
        """Compute Hamming distance between hypervectors."""
        pass
    
    @abstractmethod
    def jensen_shannon_divergence(self, hv1: Any, hv2: Any) -> float:
        """Jensen-Shannon divergence for probabilistic hypervectors."""
        pass
    
    @abstractmethod
    def wasserstein_distance(self, hv1: Any, hv2: Any) -> float:
        """Wasserstein distance for geometric hypervector comparison."""
        pass
    
    # Hierarchical and compositional operations
    
    @abstractmethod
    def hierarchical_bind(self, structure: Dict[str, Any]) -> Any:
        """Hierarchical binding for complex compositional structures.
        
        Args:
            structure: Nested dictionary of hypervectors
            
        Returns:
            Compositionally bound hypervector
        """
        pass
    
    @abstractmethod
    def semantic_projection(self, hv: Any, basis_hvs: List[Any]) -> List[float]:
        """Project hypervector onto semantic basis.
        
        Args:
            hv: Hypervector to project
            basis_hvs: List of basis hypervectors
            
        Returns:
            Projection coefficients
        """
        pass
    
    # Research benchmarking methods
    
    def benchmark_operation(self, operation_name: str, *args, **kwargs) -> Dict[str, float]:
        """Benchmark HDC operation with timing and memory metrics.
        
        Args:
            operation_name: Name of operation to benchmark
            *args, **kwargs: Arguments for the operation
            
        Returns:
            Performance metrics dictionary
        """
        import time
        import psutil
        import gc
        
        # Get operation method
        operation = getattr(self, operation_name)
        
        # Pre-benchmark cleanup
        gc.collect()
        
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Timing
        start_time = time.perf_counter()
        result = operation(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'execution_time_ms': (end_time - start_time) * 1000,
            'memory_delta_mb': memory_after - memory_before,
            'peak_memory_mb': memory_after,
            'operation': operation_name
        }
    
    def statistical_similarity_analysis(self, hv1: Any, hv2: Any, num_samples: int = 1000) -> Dict[str, float]:
        """Statistical analysis of hypervector similarity.
        
        Args:
            hv1, hv2: Hypervectors to analyze
            num_samples: Number of random samples for comparison
            
        Returns:
            Statistical similarity metrics
        """
        # Actual similarity
        actual_sim = self.cosine_similarity(hv1, hv2)
        
        # Random baseline distribution
        random_similarities = []
        for _ in range(num_samples):
            random_hv = self.random_hv()
            random_similarities.append(self.cosine_similarity(hv1, random_hv))
        
        if NUMPY_AVAILABLE:
            random_similarities = np.array(random_similarities)
            
            return {
                'cosine_similarity': float(actual_sim),
                'random_baseline_mean': float(np.mean(random_similarities)),
                'random_baseline_std': float(np.std(random_similarities)),
                'z_score': float((actual_sim - np.mean(random_similarities)) / np.std(random_similarities)),
                'percentile': float(np.mean(random_similarities < actual_sim) * 100),
                'significance': float(actual_sim > np.mean(random_similarities) + 2 * np.std(random_similarities))
            }
        else:
            # Fallback without numpy
            mean_sim = sum(random_similarities) / len(random_similarities)
            std_sim = (sum((x - mean_sim) ** 2 for x in random_similarities) / len(random_similarities)) ** 0.5
            
            return {
                'cosine_similarity': float(actual_sim),
                'random_baseline_mean': float(mean_sim),
                'random_baseline_std': float(std_sim),
                'z_score': float((actual_sim - mean_sim) / std_sim) if std_sim > 0 else 0.0,
                'percentile': float(sum(1 for x in random_similarities if x < actual_sim) / len(random_similarities) * 100),
                'significance': float(actual_sim > mean_sim + 2 * std_sim)
            }