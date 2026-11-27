"""Activation functions used by the ANN layers.

The coursework asks for the logistic function, ReLU and tanh. A few more are
provided (leaky ReLU, ELU, linear, softmax) so that they can be compared in the
experiments and offered to PSO when it is asked to choose the activations
itself (see src/adapter/activation_search_adapter.py).

Every activation is a small object with one method, forward(z). No derivatives
are needed because the network is trained by PSO, not by backpropagation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import numpy as np


class ActivationFunction(ABC):
    """Common interface for all activation functions."""

    #: short name used in configuration dictionaries and in result tables
    name: str = "activation"

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """Apply the function to the weighted input of a layer.

        Args:
            z: array of shape (n_samples, n_neurons).

        Returns:
            Array of the same shape.
        """
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    """Logistic function 1 / (1 + e^-z); output in (0, 1)."""

    name = "logistic"

    def forward(self, z: np.ndarray) -> np.ndarray:
        # PSO is free to push weights to large values, so z can be huge.
        # Clipping keeps np.exp from overflowing without changing the result
        # (the logistic function is already 0 or 1 to machine precision there).
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))


class Tanh(ActivationFunction):
    """Hyperbolic tangent; output in (-1, 1)."""

    name = "tanh"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)


class ReLU(ActivationFunction):
    """Rectified linear unit, max(0, z)."""

    name = "relu"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)


class LeakyReLU(ActivationFunction):
    """ReLU with a small slope for negative inputs, so a unit never goes
    completely silent."""

    name = "leaky_relu"

    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0.0, z, self.negative_slope * z)


class ELU(ActivationFunction):
    """Exponential linear unit: z for z > 0, e^z - 1 otherwise."""

    name = "elu"

    def forward(self, z: np.ndarray) -> np.ndarray:
        # np.minimum stops exp() overflowing on the branch np.where discards.
        return np.where(z > 0.0, z, np.exp(np.minimum(z, 0.0)) - 1.0)


class Linear(ActivationFunction):
    """Identity. Included mainly as a control: a hidden layer that uses it adds
    no non-linearity."""

    name = "linear"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z


class Softmax(ActivationFunction):
    """Turns each row of scores into a probability distribution. Only used on
    the output layer of a multiclass network."""

    name = "softmax"

    def forward(self, z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=1, keepdims=True)  # avoids overflow in exp
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# Name -> class lookup used by MLPBuilder. "sigmoid" is kept as an alias of
# "logistic" because both names are in common use.
ACTIVATIONS: Dict[str, Type[ActivationFunction]] = {
    "logistic": Sigmoid,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
    "linear": Linear,
    "softmax": Softmax,
}

# Activations that make sense in a hidden layer. This is the menu PSO chooses
# from when the activations are encoded in the particle.
HIDDEN_ACTIVATION_CHOICES: List[str] = ["logistic", "tanh", "relu", "leaky_relu", "elu"]


def make_activation(name: str) -> ActivationFunction:
    """Create an activation object from its configuration name."""
    try:
        return ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {sorted(ACTIVATIONS)}."
        ) from None
