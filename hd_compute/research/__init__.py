"""Advanced research algorithms for hyperdimensional computing."""

# Core novel algorithms that work without external dependencies
from .novel_algorithms import (
    AdvancedTemporalHDC,
    ConcreteAttentionHDC,
    NeurosymbolicHDC,
    TemporalHDC,
    CausalHDC,
    AttentionHDC,
    MetaLearningHDC
)

from .quantum_hdc import (
    ConcreteQuantumHDC,
    QuantumInspiredOperations,
    QuantumInspiredHDC
)

# Optional imports with graceful fallbacks
try:
    from .fractional_operations import FractionalHDC
except ImportError:
    FractionalHDC = None

try:
    from .adaptive_memory import AdaptiveHierarchicalMemory
except ImportError:
    AdaptiveHierarchicalMemory = None

try:
    from .statistical_analysis import HDCStatisticalAnalyzer
except ImportError:
    HDCStatisticalAnalyzer = None

__all__ = [
    'AdvancedTemporalHDC',
    'ConcreteAttentionHDC',
    'NeurosymbolicHDC',
    'ConcreteQuantumHDC',
    'QuantumInspiredOperations',
    'QuantumInspiredHDC',
    'TemporalHDC',
    'CausalHDC',
    'AttentionHDC',
    'MetaLearningHDC'
]

# Add optional exports if available
if FractionalHDC is not None:
    __all__.append('FractionalHDC')
if AdaptiveHierarchicalMemory is not None:
    __all__.append('AdaptiveHierarchicalMemory')
if HDCStatisticalAnalyzer is not None:
    __all__.append('HDCStatisticalAnalyzer')