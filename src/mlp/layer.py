"""A single fully connected layer."""

from typing import Optional

import numpy as np

from src.mlp.activations import ActivationFunction


class Layer:
    """Weights, biases and an activation function.

    A layer never changes its own parameters. They are always written from
    outside through set_parameters(), which is how PSO trains the network.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationFunction,
        rng: Optional[np.random.Generator] = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Starting values only matter if the layer is used before PSO has
        # written to it (e.g. when testing the evaluation code on a random
        # network), so a small random matrix is enough.
        rng = rng if rng is not None else np.random.default_rng()
        self.weights = rng.normal(scale=0.1, size=(input_size, output_size))
        self.biases = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return activation(X @ weights + biases) for a batch of row vectors."""
        z = X @ self.weights + self.biases
        return self.activation.forward(z)

    def count_parameters(self) -> int:
        """Number of weights plus number of biases."""
        return self.input_size * self.output_size + self.output_size

    def get_parameters(self) -> np.ndarray:
        """Flatten to one vector: all weights (row by row), then all biases."""
        return np.concatenate([self.weights.ravel(), self.biases.ravel()])

    def set_parameters(self, parameters: np.ndarray) -> None:
        """Inverse of get_parameters(): rebuild the weight matrix and the bias
        vector from a flat slice of a particle's position."""
        n_weights = self.input_size * self.output_size
        expected = n_weights + self.output_size
        if parameters.shape[0] != expected:
            raise ValueError(f"Layer expects {expected} parameters, got {parameters.shape[0]}.")
        self.weights = parameters[:n_weights].reshape(self.input_size, self.output_size)
        self.biases = parameters[n_weights:].reshape(1, self.output_size)
