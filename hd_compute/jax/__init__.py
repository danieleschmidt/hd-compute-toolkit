"""JAX backend for hyperdimensional computing."""

try:
    from .hdc_jax import HDComputeJAX
    __all__ = ["HDComputeJAX"]
except ImportError:
    import warnings
    warnings.warn("JAX not available. Install with: pip install jax jaxlib")
    
    class HDComputeJAX:
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX not available. Install with: pip install jax jaxlib")
    
    __all__ = ["HDComputeJAX"]