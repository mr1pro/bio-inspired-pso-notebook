"""Abstract interface PSO uses to evaluate candidate solutions (section 3.3)."""

from abc import ABC, abstractmethod

import numpy as np


class FitnessFunction(ABC):
    """Interface implemented by anything PSO is asked to minimise.

    PSO never knows what evaluate() does internally — this is what keeps
    the PSO component fully independent of the MLP component.
    """

    @abstractmethod
    def evaluate(self, position: np.ndarray) -> float:
        """Return a scalar fitness for `position`. Smaller is better (minimisation)."""
        raise NotImplementedError
