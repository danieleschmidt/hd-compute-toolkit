"""
HD-Compute-Toolkit: High-performance hyperdimensional computing library.

A comprehensive toolkit for hyperdimensional computing with PyTorch and JAX backends,
featuring optimized kernels for FPGA and Vulkan acceleration.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

from .core import HDCompute
from .torch import HDComputeTorch
from .jax import HDComputeJAX
from .memory import ItemMemory, AssociativeMemory
from .applications import SpeechCommandHDC, SemanticMemory
from .database import DatabaseConnection, ExperimentRepository, MetricsRepository, BenchmarkRepository
from .cache import CacheManager, HypervectorCache

__all__ = [
    "HDCompute", 
    "HDComputeTorch", 
    "HDComputeJAX", 
    "ItemMemory", 
    "AssociativeMemory",
    "SpeechCommandHDC",
    "SemanticMemory",
    "DatabaseConnection",
    "ExperimentRepository",
    "MetricsRepository", 
    "BenchmarkRepository",
    "CacheManager",
    "HypervectorCache"
]