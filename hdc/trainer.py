"""
HDTrainer — iterative prototype learning.

Training loop:
1. Encode all samples → HVs.
2. For each class, bundle all its HVs → initial prototype.
3. (Optional) Iterative refinement: re-classify, add misclassified samples
   to their true-class prototype, subtract them from the wrong prototype.
4. Binarise / threshold prototypes before storing in the classifier.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable

from .memory import HDClassifier
from .encoder import HDEncoder


class HDTrainer:
    """
    Trains an HDClassifier from labelled samples.

    Parameters
    ----------
    encoder : HDEncoder
        Encoder to convert raw feature vectors into HVs.
    classifier : HDClassifier
        The associative memory to train.
    n_iter : int
        Number of refinement iterations (0 = single-pass bundle only).
    verbose : bool
        Print progress during training.
    """

    def __init__(
        self,
        encoder: HDEncoder,
        classifier: HDClassifier,
        n_iter: int = 10,
        verbose: bool = False,
    ):
        self.encoder    = encoder
        self.classifier = classifier
        self.n_iter     = n_iter
        self.verbose    = verbose

    def fit(
        self,
        X: np.ndarray,
        y: List[str],
        encode_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> "HDTrainer":
        """
        Train on labelled feature matrix X and string labels y.

        Args:
            X:         2-D array (n_samples, n_features).
            y:         List of string class labels.
            encode_fn: Optional custom encode function x → HV. If None,
                       uses encoder.encode_continuous.

        Returns:
            self (for chaining).
        """
        X = np.asarray(X, dtype=float)
        y = list(y)
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples.")

        encode = encode_fn or self.encoder.encode_continuous

        # --- Pass 1: encode all samples ---
        if self.verbose:
            print(f"[HDTrainer] Encoding {len(y)} samples …")
        hvs = np.stack([encode(X[i]) for i in range(len(y))], axis=0).astype(float)

        classes = sorted(set(y))

        # --- Initial prototype: bundle per class ---
        prototypes: dict[str, np.ndarray] = {}
        for cls in classes:
            mask = np.array([yi == cls for yi in y])
            bundle = hvs[mask].sum(axis=0)
            prototypes[cls] = bundle   # keep as float accumulator

        # Store initial (thresholded) prototypes
        for cls in classes:
            self.classifier.add_class(cls, self._threshold(prototypes[cls]))

        if self.n_iter == 0:
            return self

        # --- Iterative refinement ---
        for iteration in range(self.n_iter):
            errors = 0
            new_proto = {cls: np.zeros(self.encoder.dim, dtype=float) for cls in classes}

            for i, (hv, true_cls) in enumerate(zip(hvs, y)):
                pred_cls = self.classifier.predict(hv)
                if pred_cls != true_cls:
                    errors += 1
                    # Reinforce true class, weaken predicted class
                    new_proto[true_cls]  += hv
                    new_proto[pred_cls]  -= hv

            # Apply corrections
            for cls in classes:
                prototypes[cls] += new_proto[cls]
                self.classifier.add_class(cls, self._threshold(prototypes[cls]))

            acc = 1.0 - errors / len(y)
            if self.verbose:
                print(f"[HDTrainer] Iter {iteration + 1}/{self.n_iter}: "
                      f"train_acc={acc:.4f}, errors={errors}")

            if errors == 0:
                if self.verbose:
                    print("[HDTrainer] Perfect training accuracy — stopping early.")
                break

        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _threshold(self, proto: np.ndarray) -> np.ndarray:
        """Binarise a float prototype accumulator."""
        if self.classifier.metric == "cosine":
            result = np.sign(proto)
            zeros = result == 0
            result[zeros] = np.random.choice([-1, 1], size=zeros.sum())
        else:
            result = (proto > 0).astype(float)
            zeros = proto == 0
            result[zeros] = np.random.randint(0, 2, size=zeros.sum()).astype(float)
        return result.astype(np.int8)
