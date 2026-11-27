"""Ways of choosing a particle's informants (Strategy pattern).

Algorithm 39 line 18 needs "the informants of x" but leaves open how they are
chosen. Each class below is one answer. A strategy returns k OTHER particles;
the particle itself is added to the comparison in
Particle.get_informants_best(), because the algorithm says "including x
itself".
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class InformantSelectionStrategy(ABC):
    """Interface: choose k informants for one particle."""

    name: str = "strategy"

    def __init__(self):
        self.rng: Optional[np.random.Generator] = None

    def bind(self, rng: np.random.Generator) -> None:
        """Give the strategy the swarm's random generator so that a run is
        fully determined by one seed."""
        self.rng = rng

    def begin_iteration(self, population: List) -> None:
        """Called once per PSO iteration before any selection. Strategies that
        need to look at the whole swarm can do that work here once."""

    @abstractmethod
    def select_informants(self, particle, population: List, k: int) -> List:
        """Return k distinct particles from population, never `particle`."""
        raise NotImplementedError


class RandomInformantSelection(InformantSelectionStrategy):
    """k other particles drawn uniformly at random.

    The Population calls this again every iteration, which gives the scheme
    described in the book: "a small set of particles chosen randomly each
    iteration".
    """

    name = "random"

    def select_informants(self, particle, population: List, k: int) -> List:
        if self.rng is None:
            self.rng = np.random.default_rng()
        others = [p for p in population if p is not particle]
        k = min(k, len(others))
        chosen = self.rng.choice(len(others), size=k, replace=False)
        return [others[i] for i in chosen]


class NearestNeighbourInformantSelection(InformantSelectionStrategy):
    """The k particles that are currently closest in Euclidean distance.

    Neighbourhoods are local in the search space, so a group of particles that
    sits in one region shares information mostly with itself.
    """

    name = "nearest"

    def __init__(self):
        super().__init__()
        self._distances: Optional[np.ndarray] = None
        self._index_of: Dict[int, int] = {}

    def begin_iteration(self, population: List) -> None:
        # All pairwise squared distances in one go:
        # |a - b|^2 = |a|^2 + |b|^2 - 2 a.b
        positions = np.stack([p.position for p in population])
        squared_norms = np.sum(positions ** 2, axis=1)
        distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * positions @ positions.T
        np.fill_diagonal(distances, np.inf)  # a particle is never its own informant
        self._distances = distances
        self._index_of = {id(p): i for i, p in enumerate(population)}

    def select_informants(self, particle, population: List, k: int) -> List:
        if self._distances is None or id(particle) not in self._index_of \
                or len(population) != self._distances.shape[0]:
            self.begin_iteration(population)
        row = self._distances[self._index_of[id(particle)]]
        k = min(k, len(population) - 1)
        nearest = np.argsort(row, kind="stable")[:k]
        return [population[i] for i in nearest]


class RingInformantSelection(InformantSelectionStrategy):
    """Fixed ring: the particle's neighbours by index, k/2 on each side.

    This is the classic "lbest" topology. The neighbourhoods never change and
    overlap only slightly, so good positions spread around the ring slowly.
    """

    name = "ring"

    def select_informants(self, particle, population: List, k: int) -> List:
        n = len(population)
        index = next(i for i, p in enumerate(population) if p is particle)
        k = min(k, n - 1)
        chosen: List[int] = []
        offset = 1
        while len(chosen) < k:
            for neighbour in ((index + offset) % n, (index - offset) % n):
                if neighbour != index and neighbour not in chosen and len(chosen) < k:
                    chosen.append(neighbour)
            offset += 1
        return [population[i] for i in chosen]


INFORMANT_STRATEGIES = {
    "random": RandomInformantSelection,
    "nearest": NearestNeighbourInformantSelection,
    "ring": RingInformantSelection,
}


def make_informant_strategy(name: str) -> InformantSelectionStrategy:
    """Create a strategy object from its configuration name."""
    try:
        return INFORMANT_STRATEGIES[name]()
    except KeyError:
        raise ValueError(
            f"Unknown informant strategy '{name}'. Available: {sorted(INFORMANT_STRATEGIES)}."
        ) from None
