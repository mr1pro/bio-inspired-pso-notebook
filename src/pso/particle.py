"""One particle of the swarm.

Comments of the form "[A39 L24]" give the line of Algorithm 39 (Particle Swarm
Optimization) in Luke, Essentials of Metaheuristics, 2nd ed., Section 3.5,
that the code implements.
"""

from typing import List, Optional, Tuple

import numpy as np

from src.pso.fitness_function import FitnessFunction


class Particle:
    """A candidate solution that moves through the search space.

    Attributes:
        position:       current location x (for the ANN: all weights and biases)
        velocity:       current velocity v
        fitness:        fitness of the current position
        best_position:  x*, the fittest location this particle has visited
        best_fitness:   fitness at x*
        informants:     the other particles this one currently listens to
    """

    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.fitness = -np.inf
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf
        self.informants: List["Particle"] = []

    def evaluate(self, fitness_function: FitnessFunction) -> float:
        """[A39 L13] AssessFitness(x): score the current position."""
        self.fitness = float(fitness_function.evaluate(self.position))
        return self.fitness

    def update_personal_best(self) -> None:
        """Remember the current position if it is the best so far. This keeps
        x*, the "previous fittest location of x" that [A39 L17] reads."""
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

    def get_informants_best(self) -> Tuple[np.ndarray, float]:
        """[A39 L18] x+: the fittest location found by this particle's
        informants, "including x itself" as the algorithm states. Because the
        particle is always in the comparison, x+ is never worse than x*.
        """
        best_position = self.best_position
        best_fitness = self.best_fitness
        for informant in self.informants:
            if informant.best_fitness > best_fitness:
                best_fitness = informant.best_fitness
                best_position = informant.best_position
        return best_position, best_fitness

    def update_velocity(
        self,
        informants_best: np.ndarray,
        global_best: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        rng: np.random.Generator,
        max_velocity: Optional[float] = None,
    ) -> None:
        """[A39 L20-24] Work out the new velocity.

        Args:
            informants_best: x+ from get_informants_best()            [A39 L18]
            global_best:     x!, the fittest location found by anyone [A39 L19]
            alpha: proportion of the old velocity that is kept        [A39 L2]
            beta:  pull towards the particle's own best x*            [A39 L3]
            gamma: pull towards the informants' best x+               [A39 L4]
            delta: pull towards the global best x!                    [A39 L5]
            max_velocity: optional limit on |v_i| (an extension; not in A39)
        """
        n = self.position.shape[0]

        # [A39 L20] "for each dimension i do": the loop over dimensions is done
        # by NumPy. Each dimension gets its own random numbers, as in the book.
        b = rng.uniform(0.0, beta, size=n)   # [A39 L21] b <- random in [0, beta]
        c = rng.uniform(0.0, gamma, size=n)  # [A39 L22] c <- random in [0, gamma]
        d = rng.uniform(0.0, delta, size=n)  # [A39 L23] d <- random in [0, delta]

        # [A39 L24] v_i <- alpha*v_i + b(x*_i - x_i) + c(x+_i - x_i) + d(x!_i - x_i)
        self.velocity = (
            alpha * self.velocity
            + b * (self.best_position - self.position)
            + c * (informants_best - self.position)
            + d * (global_best - self.position)
        )

        if max_velocity is not None:
            np.clip(self.velocity, -max_velocity, max_velocity, out=self.velocity)

    def update_position(self, epsilon: float = 1.0) -> None:
        """[A39 L26] x <- x + epsilon * v."""
        self.position = self.position + epsilon * self.velocity
