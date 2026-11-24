"""The complete MLP, composed of ordered Layer objects (section 2.6 of CW-design.pdf)."""

from typing import List

import numpy as np

from src.mlp.layer import Layer


class Network:
    """Generic multilayer perceptron: holds layers, does forward/predict,
    and exposes/accepts its full parameter vector for PSO to optimise.
    """

    def __init__(self):
        self.layers: List[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        """Append a layer, preserving order."""
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Pass X through every layer in order; return the final layer's output."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for forward(); callers apply their own decision rule
        (threshold for binary, argmax for multiclass) on the result.
        """
        return self.forward(X)

    def get_parameters(self) -> np.ndarray:
        """Concatenate every layer's parameters into one vector, in layer order."""
        return np.concatenate([layer.get_parameters() for layer in self.layers])

    def set_parameters(self, parameters: np.ndarray) -> None:
        """Split a full parameter vector across layers and install each slice."""
        offset = 0
        for layer in self.layers:
            n = layer.count_parameters()
            layer.set_parameters(parameters[offset:offset + n])
            offset += n

    def count_parameters(self) -> int:
        """Total dimensionality D of the PSO search space for this network."""
        return sum(layer.count_parameters() for layer in self.layers)
