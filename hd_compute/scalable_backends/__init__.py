"""Scalable HDC backends with performance optimization and concurrent processing."""

from .scalable_python import ScalableHDComputePython

__all__ = ['ScalableHDComputePython']

# Optional imports for backends that require dependencies
try:
    from .scalable_numpy import ScalableHDComputeNumPy
    __all__.append('ScalableHDComputeNumPy')
except ImportError:
    ScalableHDComputeNumPy = None

try:
    from .scalable_torch import ScalableHDComputeTorch
    __all__.append('ScalableHDComputeTorch')
except ImportError:
    ScalableHDComputeTorch = None