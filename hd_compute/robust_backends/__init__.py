"""Robust HDC backends with comprehensive error handling and validation."""

from .robust_python import RobustHDComputePython

__all__ = ['RobustHDComputePython']

# Optional imports for backends that require dependencies
try:
    from .robust_numpy import RobustHDComputeNumPy
    __all__.append('RobustHDComputeNumPy')
except ImportError:
    RobustHDComputeNumPy = None

try:
    from .robust_torch import RobustHDComputeTorch
    __all__.append('RobustHDComputeTorch')
except ImportError:
    RobustHDComputeTorch = None