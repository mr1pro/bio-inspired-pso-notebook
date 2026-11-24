"""Loss functions used to train/evaluate the MLP (section 2.4 of CW-design.pdf)."""

from abc import ABC, abstractmethod

import numpy as np

_EPS = 1e-12  # guards against log(0)


class Loss(ABC):
    """Common interface for all loss functions."""

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute a scalar loss value from targets and predictions."""
        raise NotImplementedError


class MSE(Loss):
    """Mean squared error."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy, for single-output sigmoid networks."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, _EPS, 1 - _EPS)
        losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.mean(losses))


class CategoricalCrossEntropy(Loss):
    """Categorical cross-entropy, for one-hot targets and softmax predictions."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, _EPS, 1 - _EPS)
        # Sum over classes, mean over samples.
        per_sample = -np.sum(y_true * np.log(y_pred), axis=1)
        return float(np.mean(per_sample))
