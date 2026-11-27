"""Builder for the ANN: turns a short configuration into a ready network."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.mlp.activations import ActivationFunction, make_activation
from src.mlp.layer import Layer
from src.mlp.losses import Loss, make_loss
from src.mlp.network import Network

_TASKS = ("binary_classification", "multiclass_classification")


class MLPBuilder:
    """Builds a Network and a matching Loss from a dataset and a configuration.

    The user chooses the hidden part of the network; the builder works out the
    rest. The number of inputs comes from the data, and the output layer and
    loss follow from the task:

        binary classification      1 logistic output,  binary cross-entropy
        multiclass classification  one softmax output per class, categorical
                                   cross-entropy

    Example:
        builder = MLPBuilder()
        builder.set_dataset(X_train, y_train)
        builder.set_configuration({
            "hidden_layers": [8, 4],            # two hidden layers
            "activations": ["tanh", "relu"],    # one activation per hidden layer
            "task": "binary_classification",
        })
        network, loss = builder.build()

    "hidden_layers": [] is allowed and gives a network with no hidden layer.
    Two optional entries override the defaults in the table above:
        "output_activation"  any activation name, e.g. "tanh"
        "loss"               "bce", "mse" or "cce"
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.config: Dict = {}
        self.rng = rng

    def set_dataset(self, X: np.ndarray, y: np.ndarray) -> "MLPBuilder":
        """Store the training data. Only its shape is used."""
        self.X = X
        self.y = y
        return self

    def set_configuration(self, config: Dict) -> "MLPBuilder":
        """Store the architecture after checking that it is consistent."""
        hidden_layers = list(config.get("hidden_layers", []))
        activations = list(config.get("activations", []))

        if len(hidden_layers) != len(activations):
            raise ValueError(
                "One activation is needed per hidden layer: got "
                f"{len(hidden_layers)} hidden layers and {len(activations)} activations."
            )
        if any(not isinstance(n, (int, np.integer)) or n <= 0 for n in hidden_layers):
            raise ValueError(f"hidden_layers must be positive integers, got {hidden_layers}.")
        for name in activations:
            make_activation(name)  # raises ValueError for an unknown name
        if "softmax" in activations:
            raise ValueError("softmax is only meant for the output layer.")
        if config.get("task") not in _TASKS:
            raise ValueError(f"config['task'] must be one of {_TASKS}.")
        if config.get("output_activation") is not None:
            make_activation(config["output_activation"])  # raises ValueError if unknown

        self.config = dict(config, hidden_layers=hidden_layers, activations=activations)
        return self

    def determine_input_size(self) -> int:
        """Number of input neurons = number of features in the data."""
        return self.X.shape[1]

    def determine_output_size(self) -> int:
        """One output for a binary task, one per class otherwise."""
        if self.config["task"] == "binary_classification":
            return 1
        if self.y.ndim == 1:
            return len(np.unique(self.y))
        return self.y.shape[1]

    def determine_output_activation(self) -> ActivationFunction:
        if self.config.get("output_activation") is not None:
            return make_activation(self.config["output_activation"])
        if self.config["task"] == "binary_classification":
            return make_activation("logistic")
        return make_activation("softmax")

    def determine_loss(self) -> Loss:
        if "loss" in self.config:
            return make_loss(self.config["loss"])
        if self.config["task"] == "binary_classification":
            return make_loss("bce")
        return make_loss("cce")

    def build_layers(self, network: Network) -> None:
        """Create the hidden layers, then the output layer, in order."""
        hidden_layers: List[int] = self.config["hidden_layers"]
        activations: List[str] = self.config["activations"]

        previous_size = self.determine_input_size()
        for size, activation_name in zip(hidden_layers, activations):
            network.add_layer(
                Layer(previous_size, int(size), make_activation(activation_name), self.rng)
            )
            previous_size = int(size)

        network.add_layer(
            Layer(
                previous_size,
                self.determine_output_size(),
                self.determine_output_activation(),
                self.rng,
            )
        )

    def build(self) -> Tuple[Network, Loss]:
        """Return (network, loss)."""
        if self.X is None or self.y is None:
            raise ValueError("Call set_dataset(X, y) before build().")
        if not self.config:
            raise ValueError("Call set_configuration(config) before build().")

        network = Network()
        self.build_layers(network)
        return network, self.determine_loss()
