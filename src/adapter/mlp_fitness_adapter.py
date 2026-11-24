"""Adapter pattern: connects an MLP to PSO's FitnessFunction interface (section 4)."""

import numpy as np

from src.mlp.losses import Loss
from src.mlp.network import Network
from src.pso.fitness_function import FitnessFunction


class MLPFitnessAdapter(FitnessFunction):
    """Makes a Network+Loss evaluable by PSO as FitnessFunction.evaluate(position).

    This is the ONLY component that knows how to turn a PSO candidate vector
    into an MLP evaluation. PSO itself never sees the network.

    IMPORTANT: uses training data only. The test set must never appear here.
    """

    def __init__(self, network: Network, loss: Loss, X_train: np.ndarray, y_train: np.ndarray):
        self.network = network
        self.loss = loss
        self.X_train = X_train
        self.y_train = y_train

    def evaluate(self, position: np.ndarray) -> float:
        self.network.set_parameters(position)
        predictions = self.network.forward(self.X_train)
        return self.loss.compute(self.y_train, predictions)
