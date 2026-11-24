"""Single layer of the MLP (section 2.5 of CW-design.pdf)."""

import numpy as np

from src.mlp.activations import ActivationFunction


class Layer:
    """One fully-connected layer: weights, biases and an activation function.

    The layer never optimises or mutates its own parameters: PSO (via the
    Network/Adapter) is solely responsible for choosing parameter values.
    """

    def __init__(self, input_size: int, output_size: int, activation: ActivationFunction):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Small random init; PSO will overwrite these via set_parameters anyway,
        # but a Layer must be usable (forward-able) immediately after construction.
        rng = np.random.default_rng()
        self.weights = rng.normal(scale=0.1, size=(input_size, output_size))
        self.biases = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute activation(X @ weights + biases)."""
        z = X @ self.weights + self.biases
        return self.activation.forward(z)

    def get_parameters(self) -> np.ndarray:
        """Flatten weights then biases into a 1-D vector (consistent ordering)."""
        return np.concatenate([self.weights.flatten(), self.biases.flatten()])

    def set_parameters(self, parameters: np.ndarray) -> None:
        """Rebuild weights/biases from a flat vector produced by get_parameters()."""
        n_weights = self.input_size * self.output_size
        w_flat = parameters[:n_weights]
        b_flat = parameters[n_weights:n_weights + self.output_size]
        self.weights = w_flat.reshape(self.input_size, self.output_size)
        self.biases = b_flat.reshape(1, self.output_size)

    def count_parameters(self) -> int:
        return self.input_size * self.output_size + self.output_size
