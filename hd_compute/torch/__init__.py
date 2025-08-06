"""PyTorch backend for hyperdimensional computing."""

try:
    from .hdc_torch import HDComputeTorch
    __all__ = ["HDComputeTorch"]
except ImportError:
    import warnings
    warnings.warn("PyTorch not available. Install with: pip install torch")
    
    class HDComputeTorch:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")
    
    __all__ = ["HDComputeTorch"]