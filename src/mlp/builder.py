"""Builder pattern: constructs a Network + Loss from hyperparameters (section 2.7)."""

from typing import Dict, List

import numpy as np

from src.mlp.activations import ActivationFunction, Sigmoid, Softmax, Tanh
from src.mlp.layer import Layer
from src.mlp.losses import BinaryCrossEntropy, CategoricalCrossEntropy, Loss
from src.mlp.network import Network

_ACTIVATION_MAP = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}


class MLPBuilder:
    """Builds a complete MLP from a dataset + a user-specified configuration.

    Example config:
        {
            "hidden_layers": [8, 5],
            "activations": ["tanh", "sigmoid"],
            "task": "multiclass_classification",  # or "binary_classification"
        }
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.config: Dict = {}

    def set_dataset(self, X: np.ndarray, y: np.ndarray) -> "MLPBuilder":
        self.X = X
        self.y = y
        return self

    def set_configuration(self, config: Dict) -> "MLPBuilder":
        hidden_layers = config.get("hidden_layers", [])
        activations = config.get("activations", [])
        if len(hidden_layers) != len(activations):
            raise ValueError(
                "Number of activation functions must match number of hidden layers: "
                f"{len(hidden_layers)} hidden layers vs {len(activations)} activations."
            )
        if any(not isinstance(n, int) or n <= 0 for n in hidden_layers):
            raise ValueError(f"hidden_layers must be positive integers, got {hidden_layers}.")
        unknown = [a for a in activations if a not in _ACTIVATION_MAP]
        if unknown:
            raise ValueError(
                f"Unknown activation name(s) {unknown}; must be one of {sorted(_ACTIVATION_MAP)}."
            )
        if config.get("task") not in ("binary_classification", "multiclass_classification"):
            raise ValueError(
                "config['task'] must be 'binary_classification' or 'multiclass_classification' "
                "(must be stated explicitly, not inferred from the labels)."
            )
        self.config = config
        return self

    def determine_input_size(self) -> int:
        return self.X.shape[1]

    def determine_output_size(self) -> int:
        if self.config["task"] == "binary_classification":
            return 1
        if self.y.ndim == 1:
            # Use the count of distinct labels actually present, not max(y)+1,
            # so a split that happens to miss the globally-largest class label
            # doesn't silently undersize the output layer.
            return len(np.unique(self.y))
        return self.y.shape[1]

    def determine_output_activation(self) -> ActivationFunction:
        if self.config["task"] == "binary_classification":
            return Sigmoid()
        return Softmax()

    def determine_loss(self) -> Loss:
        if self.config["task"] == "binary_classification":
            return BinaryCrossEntropy()
        return CategoricalCrossEntropy()

    def build_layers(self, network: Network) -> None:
        hidden_layers: List[int] = self.config.get("hidden_layers", [])
        activations: List[str] = self.config.get("activations", [])

        input_size = self.determine_input_size()
        prev_size = input_size

        for size, activation_name in zip(hidden_layers, activations):
            activation = _ACTIVATION_MAP[activation_name]()
            network.add_layer(Layer(prev_size, size, activation))
            prev_size = size

        output_size = self.determine_output_size()
        output_activation = self.determine_output_activation()
        network.add_layer(Layer(prev_size, output_size, output_activation))

    def build(self):
        """Coordinate construction; return (network, loss)."""
        if self.X is None or self.y is None:
            raise ValueError("Call set_dataset(X, y) before build().")
        if not self.config:
            raise ValueError("Call set_configuration(config) before build().")

        network = Network()
        self.build_layers(network)
        loss = self.determine_loss()
        return network, loss
