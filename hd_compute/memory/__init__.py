"""Memory structures for hyperdimensional computing."""

try:
    from .item_memory import ItemMemory
    from .associative_memory import AssociativeMemory
    __all__ = ["ItemMemory", "AssociativeMemory"]
except ImportError:
    # Fallback to simple implementations
    from .simple_memory import SimpleItemMemory as ItemMemory
    from .simple_memory import SimpleAssociativeMemory as AssociativeMemory
    __all__ = ["ItemMemory", "AssociativeMemory"]