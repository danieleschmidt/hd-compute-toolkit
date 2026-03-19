"""
HDEncoder — encodes raw data into hyperdimensional vectors.

Three encoding strategies:
1. **Random-projection encoding** (continuous features): projects a feature
   vector into HD space via a fixed random matrix; robust, preserves metric
   structure (Johnson-Lindenstrauss).

2. **Level encoding** (integers / ordinal values): a codebook of D-dim HVs
   is built such that adjacent levels share ~50% bits, distant levels are
   nearly orthogonal. Encodes scalars by indexing the codebook.

3. **Positional (sequence) encoding**: encodes ordered sequences by binding
   each element's HV with a position HV, then bundling. Captures both
   identity and position.
"""

import numpy as np
from typing import List, Optional, Union


class HDEncoder:
    """
    Hyperdimensional encoder.

    Parameters
    ----------
    dim : int
        Dimensionality of hypervectors (default 10 000).
    bipolar : bool
        If True use ±1 vectors; if False use binary 0/1 vectors.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, dim: int = 10_000, bipolar: bool = True, seed: Optional[int] = None):
        self.dim = dim
        self.bipolar = bipolar
        self._rng = np.random.default_rng(seed)
        self._proj_matrix: Optional[np.ndarray] = None
        self._level_codebook: Optional[np.ndarray] = None
        self._pos_codebook: Optional[np.ndarray] = None
        self._item_memory: dict = {}   # symbol → HV cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def random_hv(self) -> np.ndarray:
        """Generate a single random hypervector."""
        if self.bipolar:
            return self._rng.choice([-1, 1], size=self.dim).astype(np.int8)
        else:
            return self._rng.integers(0, 2, size=self.dim, dtype=np.int8)

    def encode_continuous(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a continuous feature vector via random projection.

        The same projection matrix is reused across calls, so the HD space
        is consistent. The result is binarised to ±1 (bipolar) or 0/1 (binary).

        Args:
            x: 1-D float array of length n_features.

        Returns:
            Hypervector of length self.dim.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.shape[0]
        if self._proj_matrix is None or self._proj_matrix.shape[1] != n:
            self._proj_matrix = self._rng.standard_normal((self.dim, n))
        projected = self._proj_matrix @ x
        if self.bipolar:
            return np.sign(projected).astype(np.int8)
        else:
            return (projected > 0).astype(np.int8)

    def encode_level(self, value: int, n_levels: int) -> np.ndarray:
        """
        Encode an integer value using a level codebook.

        Adjacent levels share ~50% flipped bits; the codebook provides a
        smooth, graded encoding for ordinal data.

        Args:
            value:    Integer in [0, n_levels).
            n_levels: Total number of distinct levels.

        Returns:
            Hypervector of length self.dim.
        """
        if self._level_codebook is None or self._level_codebook.shape[0] != n_levels:
            self._level_codebook = self._build_level_codebook(n_levels)
        value = int(np.clip(value, 0, n_levels - 1))
        return self._level_codebook[value].copy()

    def encode_sequence(self, symbols: List[str]) -> np.ndarray:
        """
        Encode an ordered sequence of symbols using positional encoding.

        Each symbol is bound with its position HV and all positions are
        bundled together (majority vote / sum). The result captures both
        what is present and where.

        Args:
            symbols: List of hashable symbols (e.g. tokens, characters).

        Returns:
            Hypervector of length self.dim representing the sequence.
        """
        if not symbols:
            return np.zeros(self.dim, dtype=np.int8)

        max_pos = len(symbols)
        pos_hvs = self._get_position_hvs(max_pos)

        bound = []
        for i, sym in enumerate(symbols):
            sym_hv  = self._get_item_hv(sym)
            pos_hv  = pos_hvs[i]
            bound.append(self._bind(sym_hv, pos_hv))

        return self._bundle_threshold(bound)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_level_codebook(self, n_levels: int) -> np.ndarray:
        """
        Build a codebook of n_levels HVs with graded similarity.

        Level 0 is a fresh random HV. Each successive level flips
        D/(2*(n_levels-1)) bits, so level 0 and level n_levels-1 share ~0%.
        """
        codebook = np.empty((n_levels, self.dim), dtype=np.int8)
        base = self.random_hv()
        codebook[0] = base

        if n_levels == 1:
            return codebook

        # Total bits to flip going from level 0 to level n_levels-1: D/2
        flips_per_step = max(1, self.dim // (2 * (n_levels - 1)))
        indices = self._rng.permutation(self.dim)
        flip_ptr = 0

        for lvl in range(1, n_levels):
            codebook[lvl] = codebook[lvl - 1].copy()
            to_flip = indices[flip_ptr: flip_ptr + flips_per_step]
            if self.bipolar:
                codebook[lvl][to_flip] *= -1
            else:
                codebook[lvl][to_flip] ^= 1
            flip_ptr = (flip_ptr + flips_per_step) % self.dim

        return codebook

    def _get_position_hvs(self, n: int) -> List[np.ndarray]:
        """Return (or extend) position hypervectors."""
        if self._pos_codebook is None or len(self._pos_codebook) < n:
            needed = n if self._pos_codebook is None else n - len(self._pos_codebook)
            new_hvs = [self.random_hv() for _ in range(needed)]
            if self._pos_codebook is None:
                self._pos_codebook = new_hvs
            else:
                self._pos_codebook.extend(new_hvs)
        return self._pos_codebook[:n]

    def _get_item_hv(self, symbol: str) -> np.ndarray:
        """Return (or create) a random HV for a symbol."""
        if symbol not in self._item_memory:
            self._item_memory[symbol] = self.random_hv()
        return self._item_memory[symbol]

    def _bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        if self.bipolar:
            return (hv1 * hv2).astype(np.int8)
        else:
            return np.bitwise_xor(hv1, hv2)

    def _bundle_threshold(self, hvs: List[np.ndarray]) -> np.ndarray:
        stack = np.stack(hvs, axis=0).astype(float)
        summed = stack.sum(axis=0)
        if self.bipolar:
            result = np.sign(summed)
            zeros = result == 0
            result[zeros] = self._rng.choice([-1, 1], size=zeros.sum())
        else:
            n = len(hvs)
            result = np.where(summed > n / 2, 1,
                     np.where(summed < n / 2, 0,
                     self._rng.integers(0, 2, size=summed.shape)))
        return result.astype(np.int8)
