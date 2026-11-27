"""Extension: let PSO choose the hidden-layer activation functions as well.

The particle is made a little longer. For a network with D weights and biases
and H hidden layers the position has D + H numbers:

    [ w_1 ... w_D | g_1 ... g_H ]
       weights       one "activation gene" per hidden layer

A gene is a real number like every other coordinate, so PSO moves it with the
same velocity rule and needs no changes. It is turned into a choice by cutting
its range into equal bins, one per candidate activation:

    range [low, high] with choices [logistic, tanh, relu]
    low ........|........|........ high
       logistic    tanh     relu

Values outside the range count as the nearest end bin. The length of the
vector is still fixed, so plain (fixed-length) PSO is enough.
"""

from typing import List, Sequence, Tuple

import numpy as np

from src.adapter.mlp_fitness_adapter import MLPFitnessAdapter
from src.mlp.activations import HIDDEN_ACTIVATION_CHOICES, make_activation
from src.mlp.losses import Loss
from src.mlp.network import Network


class ActivationSearchAdapter(MLPFitnessAdapter):
    """Fitness adapter whose particles carry weights AND activation choices.

    Args:
        choices:    names of the activations PSO may pick from.
        gene_range: (low, high) interval that is divided into bins. It should
                    match the range the swarm is initialised in, so that every
                    choice starts out equally likely.
        (other arguments as for MLPFitnessAdapter)
    """

    def __init__(
        self,
        network: Network,
        loss: Loss,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fitness: str = "accuracy",
        choices: Sequence[str] = tuple(HIDDEN_ACTIVATION_CHOICES),
        gene_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__(network, loss, X_train, y_train, fitness)
        if len(choices) < 2:
            raise ValueError("Give PSO at least two activations to choose from.")
        self.choices: List[str] = list(choices)
        self.gene_range = (float(gene_range[0]), float(gene_range[1]))
        self.num_weights = network.count_parameters()
        self.num_genes = len(network.layers) - 1  # one per hidden layer
        if self.num_genes < 1:
            raise ValueError("The network has no hidden layer to choose an activation for.")
        # PSO must be told about the longer vector.
        self.dimension = self.num_weights + self.num_genes
        # One object per choice, reused for every evaluation.
        self._activation_objects = [make_activation(name) for name in self.choices]

    def decode_choices(self, position: np.ndarray) -> List[int]:
        """Map the activation genes at the end of the vector to bin indices."""
        low, high = self.gene_range
        genes = np.asarray(position[self.num_weights:], dtype=float)
        unit = np.clip((genes - low) / (high - low), 0.0, 1.0)
        indices = np.minimum((unit * len(self.choices)).astype(int), len(self.choices) - 1)
        return [int(i) for i in indices]

    def decode_activations(self, position: np.ndarray) -> List[str]:
        """Names of the activations a particle encodes, one per hidden layer."""
        return [self.choices[i] for i in self.decode_choices(position)]

    def install(self, position: np.ndarray) -> Network:
        """Set both the activations and the weights from a particle."""
        indices = self.decode_choices(position)
        self.network.set_hidden_activations([self._activation_objects[i] for i in indices])
        self.network.set_parameters(position[:self.num_weights])
        return self.network
