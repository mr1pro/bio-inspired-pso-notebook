"""Coupling between PSO and the ANN (Adapter pattern).

PSO wants an object with evaluate(position) -> float. The ANN offers
set_parameters(), forward() and a loss. MLPFitnessAdapter puts the first on
top of the second, and it is the only class that knows about both sides.

How a particle becomes a network:
    1. the particle's position is a vector of D real numbers, where D is the
       number of weights and biases in the network;
    2. Network.set_parameters() cuts that vector into the weight matrix and
       bias vector of each layer;
    3. the network classifies the TRAINING set;
    4. the resulting score is returned to PSO as the particle's fitness.
"""

import numpy as np

from src.mlp.losses import Loss
from src.mlp.network import Network
from src.pso.fitness_function import FitnessFunction

FITNESS_MEASURES = ("accuracy", "loss")


class MLPFitnessAdapter(FitnessFunction):
    """Lets PSO evaluate a network through the FitnessFunction interface.

    Args:
        network:  the ANN whose weights are being optimised.
        loss:     loss object, used when fitness == "loss".
        X_train:  training inputs, shape (n_samples, n_features).
        y_train:  training targets; shape (n_samples, 1) of 0/1 for a binary
                  task, or one-hot rows for a multiclass task.
        fitness:  "accuracy": fraction of training samples classified
                              correctly. This is what the coursework asks for.
                  "loss":     minus the loss (PSO maximises, so the sign is
                              flipped). A smoother alternative.

    Only training data ever reaches this class. The test set is used once,
    after PSO has finished, to measure generalisation.
    """

    def __init__(
        self,
        network: Network,
        loss: Loss,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fitness: str = "accuracy",
    ):
        if fitness not in FITNESS_MEASURES:
            raise ValueError(f"fitness must be one of {FITNESS_MEASURES}.")
        self.network = network
        self.loss = loss
        self.X_train = X_train
        self.y_train = y_train
        self.fitness = fitness
        self.dimension = network.count_parameters()
        self.evaluations = 0  # how many candidate networks have been scored

        # Integer class labels, used to compute accuracy.
        if y_train.ndim == 2 and y_train.shape[1] > 1:
            self._labels = np.argmax(y_train, axis=1)
        else:
            self._labels = y_train.ravel().astype(int)

    # -- helpers --------------------------------------------------------
    @staticmethod
    def to_labels(outputs: np.ndarray) -> np.ndarray:
        """Turn network outputs into class labels: threshold at 0.5 for one
        output unit, arg-max for several."""
        if outputs.ndim == 2 and outputs.shape[1] > 1:
            return np.argmax(outputs, axis=1)
        return (outputs.ravel() >= 0.5).astype(int)

    def install(self, position: np.ndarray) -> Network:
        """Write a particle's position into the network and return it."""
        self.network.set_parameters(position)
        return self.network

    # -- FitnessFunction ------------------------------------------------
    def evaluate(self, position: np.ndarray) -> float:
        """Fitness of one particle (higher is better)."""
        self.evaluations += 1
        self.install(position)
        outputs = self.network.forward(self.X_train)
        if self.fitness == "accuracy":
            return float(np.mean(self.to_labels(outputs) == self._labels))
        return -self.loss.compute(self.y_train, outputs)
