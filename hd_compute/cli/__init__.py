"""Command-line interface for HD-Compute-Toolkit."""

from .benchmark import benchmark
from .train import train
from .evaluate import evaluate

__all__ = ["benchmark", "train", "evaluate"]