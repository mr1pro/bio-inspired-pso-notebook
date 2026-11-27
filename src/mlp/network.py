"""The feedforward network: an ordered list of layers."""

from typing import List, Sequence

import numpy as np

from src.mlp.activations import ActivationFunction
from src.mlp.layer import Layer


class Network:
    """Multilayer perceptron with any number of layers.

    Besides the forward pass, the class can hand out all of its weights and
    biases as one flat vector and take them back in the same order. That
    vector is what a PSO particle's position represents.

    Order of the flat vector:
        [layer 1 weights, layer 1 biases, layer 2 weights, layer 2 biases, ...]
    """

    def __init__(self):
        self.layers: List[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        """Append a layer. It must accept as many inputs as the previous layer
        produces."""
        if self.layers and self.layers[-1].output_size != layer.input_size:
            raise ValueError(
                f"Layer takes {layer.input_size} inputs but the previous layer "
                f"has {self.layers[-1].output_size} outputs."
            )
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Feed X through every layer in turn and return the last output."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Network output for X (probabilities for a classification network).
        Turning these into class labels is left to the caller."""
        return self.forward(X)

    def count_parameters(self) -> int:
        """Total number of weights and biases, D = sum_l (n_{l-1} * n_l + n_l).
        This is the length of the vector PSO optimises."""
        return sum(layer.count_parameters() for layer in self.layers)

    def get_parameters(self) -> np.ndarray:
        """All weights and biases as one vector, layer by layer."""
        return np.concatenate([layer.get_parameters() for layer in self.layers])

    def set_parameters(self, parameters: np.ndarray) -> None:
        """Write a flat parameter vector back into the layers."""
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape[0] != self.count_parameters():
            raise ValueError(
                f"Network expects {self.count_parameters()} parameters, "
                f"got {parameters.shape[0]}."
            )
        offset = 0
        for layer in self.layers:
            n = layer.count_parameters()
            layer.set_parameters(parameters[offset:offset + n])
            offset += n

    # ------------------------------------------------------------------
    # Used when PSO is also asked to choose the hidden activations.
    # ------------------------------------------------------------------
    def set_hidden_activations(self, activations: Sequence[ActivationFunction]) -> None:
        """Replace the activation of every hidden layer (all but the last)."""
        hidden = self.layers[:-1]
        if len(activations) != len(hidden):
            raise ValueError(
                f"Network has {len(hidden)} hidden layers, got {len(activations)} activations."
            )
        for layer, activation in zip(hidden, activations):
            layer.activation = activation

    def describe(self) -> str:
        """One-line summary such as '4-8(tanh)-1(logistic)'."""
        if not self.layers:
            return "(empty network)"
        parts = [str(self.layers[0].input_size)]
        parts += [f"{layer.output_size}({layer.activation.name})" for layer in self.layers]
        return "-".join(parts)
