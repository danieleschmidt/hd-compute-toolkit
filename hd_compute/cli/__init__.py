"""Command-line interface for HD-Compute-Toolkit."""

from .main import main

try:
    from .benchmark import benchmark
    from .train import train 
    from .evaluate import evaluate
    __all__ = ["main", "benchmark", "train", "evaluate"]
except ImportError:
    __all__ = ["main"]