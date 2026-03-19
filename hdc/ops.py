"""
Fundamental HDC operations: Bundle and Bind.

In hyperdimensional computing, two operations are foundational:

- **Bundle** (superposition): combines multiple HVs into one that is similar
  to all of them. Implemented as element-wise majority vote (binary) or
  element-wise addition (bipolar/integer).

- **Bind**: combines two HVs into one that is dissimilar to both inputs yet
  invertible. Implemented as element-wise XOR (binary) or multiplication
  (bipolar ±1). Bind is its own inverse for binary: bind(bind(a,b), b) == a.
"""

import numpy as np
from typing import List


class BundleOperation:
    """
    Bundle (superposition) of hypervectors.

    For binary vectors: majority vote (ties broken randomly).
    For bipolar/integer vectors: element-wise sum, then binarised on request.
    """

    def __call__(self, hvs: List[np.ndarray], threshold: bool = False) -> np.ndarray:
        return self.bundle(hvs, threshold=threshold)

    @staticmethod
    def bundle(hvs: List[np.ndarray], threshold: bool = False) -> np.ndarray:
        """
        Bundle a list of hypervectors.

        Args:
            hvs: List of 1-D arrays (binary 0/1 or bipolar ±1 or integer).
            threshold: If True, binarise the result to ±1 (for bipolar) or
                       0/1 (for binary) after summing.

        Returns:
            Bundled hypervector (float sum, or thresholded).
        """
        if not hvs:
            raise ValueError("Cannot bundle an empty list.")
        stack = np.stack(hvs, axis=0)          # (n, D)
        summed = stack.sum(axis=0, dtype=float)

        if not threshold:
            return summed

        # Detect mode: bipolar (values ±1) or binary (values 0/1)
        sample = hvs[0]
        if np.all((sample == 0) | (sample == 1)):
            # Binary: majority vote; ties → random 0/1
            n = len(hvs)
            mid = n / 2.0
            result = np.where(summed > mid, 1,
                     np.where(summed < mid, 0,
                     np.random.randint(0, 2, size=summed.shape)))
            return result.astype(np.int8)
        else:
            # Bipolar: sign; zeros → random ±1
            result = np.sign(summed)
            zeros = result == 0
            result[zeros] = np.random.choice([-1, 1], size=zeros.sum())
            return result.astype(np.int8)


class BindOperation:
    """
    Bind two hypervectors into a single HV dissimilar to either.

    Binary  vectors (0/1): element-wise XOR.
    Bipolar vectors (±1) : element-wise multiplication.
    Auto-detected from the dtype / values of the inputs.
    """

    def __call__(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        return self.bind(hv1, hv2)

    @staticmethod
    def bind(hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors.

        Binding is its own inverse for both modes:
          binary:  bind(bind(a, b), b) == a
          bipolar: bind(bind(a, b), b) == a  (since (±1)²=1)

        Args:
            hv1: First hypervector.
            hv2: Second hypervector (same dimension and type as hv1).

        Returns:
            Bound hypervector.
        """
        if hv1.shape != hv2.shape:
            raise ValueError(f"Shape mismatch: {hv1.shape} vs {hv2.shape}")

        # Detect binary vs bipolar
        if np.all((hv1 == 0) | (hv1 == 1)):
            return np.bitwise_xor(hv1.astype(np.int8), hv2.astype(np.int8))
        else:
            return (hv1 * hv2).astype(np.int8)

    @staticmethod
    def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Recover original HV: unbind(bind(a,b), b) == a."""
        return BindOperation.bind(bound, key)
