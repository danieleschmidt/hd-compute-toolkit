"""
HD-Compute-Toolkit: High-performance hyperdimensional computing library.

A comprehensive toolkit for hyperdimensional computing with PyTorch and JAX backends,
featuring optimized kernels for FPGA and Vulkan acceleration.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

from .core import HDCompute
from .pure_python import HDComputePython

# Try to import NumPy backend if available
try:
    from .numpy import HDComputeNumPy
except ImportError:
    HDComputeNumPy = None

# Optional imports with fallback
try:
    from .torch import HDComputeTorch
except ImportError:
    HDComputeTorch = None

try:
    from .jax import HDComputeJAX
except ImportError:
    HDComputeJAX = None

try:
    from .memory import ItemMemory, AssociativeMemory
except ImportError:
    ItemMemory = AssociativeMemory = None

try:
    from .applications import SpeechCommandHDC, SemanticMemory
except ImportError:
    SpeechCommandHDC = SemanticMemory = None

try:
    from .database import DatabaseConnection, ExperimentRepository, MetricsRepository, BenchmarkRepository
except ImportError:
    DatabaseConnection = ExperimentRepository = MetricsRepository = BenchmarkRepository = None

try:
    from .cache import CacheManager, HypervectorCache
except ImportError:
    CacheManager = HypervectorCache = None

# Base exports always available
__all__ = [
    "HDCompute", 
    "HDComputePython",
]

# Add NumPy backend if available
if HDComputeNumPy is not None:
    __all__.append("HDComputeNumPy")

# Add optional exports if available
if HDComputeTorch is not None:
    __all__.append("HDComputeTorch")
if HDComputeJAX is not None:
    __all__.append("HDComputeJAX")
if ItemMemory is not None:
    __all__.extend(["ItemMemory", "AssociativeMemory"])
if SpeechCommandHDC is not None:
    __all__.extend(["SpeechCommandHDC", "SemanticMemory"])
if DatabaseConnection is not None:
    __all__.extend(["DatabaseConnection", "ExperimentRepository", "MetricsRepository", "BenchmarkRepository"])
if CacheManager is not None:
    __all__.extend(["CacheManager", "HypervectorCache"])