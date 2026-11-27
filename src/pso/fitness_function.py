"""The only thing PSO knows about the problem it is solving."""

from abc import ABC, abstractmethod

import numpy as np


class FitnessFunction(ABC):
    """Interface for anything PSO is asked to optimise.

    PSO calls evaluate(position) and gets a number back. It has no idea
    whether that number came from a neural network or from a test function,
    which is what keeps the PSO package free of any ANN code.

    Convention: HIGHER IS BETTER. Algorithm 39 in Essentials of Metaheuristics
    is written as a maximiser (line 14: "Fitness(x) > Fitness(Best)") and the
    coursework uses classification accuracy as the fitness, so the same
    convention is used here. A quantity that should be minimised, such as a
    loss, is simply returned with its sign flipped.
    """

    @abstractmethod
    def evaluate(self, position: np.ndarray) -> float:
        """Return the fitness of one candidate solution (higher is better)."""
        raise NotImplementedError
