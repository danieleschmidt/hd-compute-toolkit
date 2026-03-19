"""
hdc — Hyperdimensional Computing Toolkit
=========================================
Encode data into high-dimensional binary/bipolar vectors and classify
using associative memory. Inspired by Kanerva's Sparse Distributed Memory
and Plate's Holographic Reduced Representations.
"""

from .encoder import HDEncoder
from .memory import HDClassifier
from .ops import BundleOperation, BindOperation
from .trainer import HDTrainer

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "HDEncoder",
    "HDClassifier",
    "BundleOperation",
    "BindOperation",
    "HDTrainer",
]
