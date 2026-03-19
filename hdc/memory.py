"""
HDClassifier — associative memory for HD classification.

Class prototypes are stored as hypervectors. Classification is performed
by computing similarity (cosine or Hamming) between a query HV and all
stored prototypes and returning the closest class.

This mirrors biological associative memory: a partial or noisy pattern
activates the closest stored memory.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List


class HDClassifier:
    """
    Associative memory classifier for hyperdimensional computing.

    Each class has a *prototype* HV stored in memory. To classify a query
    HV, find the prototype with maximum cosine similarity (bipolar) or
    minimum Hamming distance (binary).

    Parameters
    ----------
    dim : int
        Dimensionality of hypervectors.
    metric : str
        'cosine' (default, bipolar-friendly) or 'hamming' (binary-friendly).
    """

    def __init__(self, dim: int = 10_000, metric: str = "cosine"):
        self.dim = dim
        if metric not in ("cosine", "hamming"):
            raise ValueError(f"metric must be 'cosine' or 'hamming', got {metric!r}")
        self.metric = metric
        self._prototypes: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def add_class(self, label: str, hv: np.ndarray) -> None:
        """Store or overwrite a class prototype."""
        if hv.shape != (self.dim,):
            raise ValueError(f"Expected HV of shape ({self.dim},), got {hv.shape}")
        self._prototypes[label] = hv.astype(float)

    def update_class(self, label: str, hv: np.ndarray, weight: float = 1.0) -> None:
        """
        Incrementally update an existing prototype by weighted addition.

        Useful for online training: bundle new evidence into the prototype.
        """
        if label not in self._prototypes:
            self.add_class(label, hv)
        else:
            self._prototypes[label] += weight * hv.astype(float)

    def classes(self) -> List[str]:
        """Return sorted list of known class labels."""
        return sorted(self._prototypes.keys())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def similarity(self, query: np.ndarray) -> Dict[str, float]:
        """
        Compute similarity between query and all stored prototypes.

        Returns a dict {label: score} where higher score = more similar.
        For cosine: score ∈ [-1, 1].
        For hamming: score = 1 - normalised_hamming_distance ∈ [0, 1].
        """
        q = query.astype(float)
        scores: Dict[str, float] = {}
        for label, proto in self._prototypes.items():
            if self.metric == "cosine":
                denom = np.linalg.norm(q) * np.linalg.norm(proto)
                scores[label] = float(np.dot(q, proto) / denom) if denom > 0 else 0.0
            else:  # hamming
                # Convert float prototype to binary for Hamming distance
                p_bin = (proto > 0).astype(int)
                q_bin = (q    > 0).astype(int)
                dist = np.sum(p_bin != q_bin) / self.dim
                scores[label] = 1.0 - float(dist)
        return scores

    def predict(self, query: np.ndarray) -> str:
        """Return the label of the most similar prototype."""
        if not self._prototypes:
            raise RuntimeError("No class prototypes stored. Train first.")
        scores = self.similarity(query)
        return max(scores, key=scores.__getitem__)

    def predict_proba(self, query: np.ndarray) -> Dict[str, float]:
        """
        Return softmax-normalised similarity scores as pseudo-probabilities.
        """
        scores = self.similarity(query)
        vals = np.array(list(scores.values()))
        # Shift for numerical stability then softmax
        vals -= vals.max()
        exp_vals = np.exp(vals)
        exp_vals /= exp_vals.sum()
        return dict(zip(scores.keys(), exp_vals.tolist()))

    # ------------------------------------------------------------------
    # Bulk inference
    # ------------------------------------------------------------------

    def predict_batch(self, queries: np.ndarray) -> List[str]:
        """
        Classify a 2-D array of HVs (n_samples × dim) efficiently.
        """
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"Expected shape (n, {self.dim}), got {queries.shape}")
        labels = list(self._prototypes.keys())
        # Stack prototypes: (n_classes, dim)
        P = np.stack([self._prototypes[l] for l in labels], axis=0).astype(float)
        Q = queries.astype(float)

        if self.metric == "cosine":
            # (n_samples, n_classes)
            numer = Q @ P.T
            q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
            p_norms = np.linalg.norm(P, axis=1, keepdims=True).T
            denom = q_norms * p_norms
            denom = np.where(denom == 0, 1e-8, denom)
            scores = numer / denom
        else:
            P_bin = (P > 0).astype(float)
            Q_bin = (Q > 0).astype(float)
            # Hamming similarity: 1 - mean(disagreements)
            scores = 1.0 - (Q_bin @ (1 - P_bin).T + (1 - Q_bin) @ P_bin.T) / self.dim

        best_idx = scores.argmax(axis=1)
        return [labels[i] for i in best_idx]
