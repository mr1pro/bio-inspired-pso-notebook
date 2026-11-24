"""Strategy pattern for informant selection (section 3.5)."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class InformantSelectionStrategy(ABC):
    """Abstract interface: pick k informants for a particle from the population."""

    @abstractmethod
    def select_informants(self, particle, population: List, k: int) -> List:
        """Return k distinct particles from `population`, excluding `particle` itself."""
        raise NotImplementedError


class RandomInformantSelection(InformantSelectionStrategy):
    """Selects k informants uniformly at random (excluding self)."""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    def select_informants(self, particle, population: List, k: int) -> List:
        candidates = [p for p in population if p is not particle]
        k = min(k, len(candidates))
        chosen_idx = self.rng.choice(len(candidates), size=k, replace=False)
        return [candidates[i] for i in chosen_idx]


class NearestNeighbourInformantSelection(InformantSelectionStrategy):
    """Selects the k particles closest in Euclidean distance (excluding self)."""

    def select_informants(self, particle, population: List, k: int) -> List:
        candidates = [p for p in population if p is not particle]
        distances = [
            (np.linalg.norm(particle.position - other.position), other)
            for other in candidates
        ]
        distances.sort(key=lambda pair: pair[0])
        k = min(k, len(distances))
        return [other for _, other in distances[:k]]
