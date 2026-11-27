"""Loss functions.

PSO needs no gradients, so a loss is only ever used as a number that says how
good a set of weights is. The main experiments use classification accuracy as
the fitness (as the coursework asks); the losses here give the alternative,
smoother fitness measures compared in the "fitness function" experiment.
"""

from abc import ABC, abstractmethod

import numpy as np

_EPS = 1e-12  # keeps log() away from zero


class Loss(ABC):
    """Common interface for all loss functions."""

    name: str = "loss"

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return one number: the average loss over all samples."""
        raise NotImplementedError


class MSE(Loss):
    """Mean squared error."""

    name = "mse"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy for a single logistic output unit."""

    name = "bce"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, _EPS, 1.0 - _EPS)
        per_sample = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(np.mean(per_sample))


class CategoricalCrossEntropy(Loss):
    """Categorical cross-entropy for one-hot targets and softmax outputs."""

    name = "cce"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, _EPS, 1.0 - _EPS)
        per_sample = -np.sum(y_true * np.log(y_pred), axis=1)
        return float(np.mean(per_sample))


LOSSES = {
    "mse": MSE,
    "bce": BinaryCrossEntropy,
    "cce": CategoricalCrossEntropy,
}


def make_loss(name: str) -> Loss:
    """Create a loss object from its configuration name."""
    try:
        return LOSSES[name]()
    except KeyError:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(LOSSES)}.") from None
