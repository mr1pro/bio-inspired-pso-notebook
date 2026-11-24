"""Activation functions used by MLP layers (section 2.3 of CW-design.pdf)."""

from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    """Common interface for all activation functions."""

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """Apply the activation function element-wise (or row-wise for softmax).

        Args:
            z: weighted input of a layer, shape (n_samples, n_neurons).

        Returns:
            Transformed values, same shape as z (except Softmax keeps shape too).
        """
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    """Sigmoid activation: values in (0, 1)."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow in np.exp for very negative/positive z.
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))


class Tanh(ActivationFunction):
    """Hyperbolic tangent activation: values in (-1, 1)."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)


class Softmax(ActivationFunction):
    """Softmax activation: converts scores into a per-row probability distribution."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        # Subtract row-wise max for numerical stability.
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
