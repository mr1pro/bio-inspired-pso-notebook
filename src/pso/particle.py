"""A single candidate solution in the PSO search space (section 3.4)."""

from typing import List, Optional, Tuple

import numpy as np

from src.pso.fitness_function import FitnessFunction


class Particle:
    """One particle: position, velocity, personal best, and its informants."""

    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_fitness = np.inf
        self.fitness = np.inf
        self.informants: List["Particle"] = []

    def evaluate(self, fitness_function: FitnessFunction) -> float:
        """Evaluate current position; store and return the fitness."""
        self.fitness = fitness_function.evaluate(self.position)
        return self.fitness

    def update_personal_best(self) -> None:
        """If the current position is better than the stored best, replace it."""
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

    def get_informants_best(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (position, fitness) of the best-performing informant.

        p_I = argmin_{k in informants} f(p_best_k)
        """
        best_position = None
        best_fitness = np.inf
        for informant in self.informants:
            if informant.best_fitness < best_fitness:
                best_fitness = informant.best_fitness
                best_position = informant.best_position
        return best_position, best_fitness

    def update_velocity(
        self,
        informants_best: Optional[np.ndarray],
        global_best: Optional[np.ndarray],
        inertia_weight: float,
        cognitive_coefficient: float,
        social_coefficient: float,
        global_coefficient: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """v(t+1) = alpha*v(t) + beta*r1*(p_best - x) + delta*r2*(p_I - x) + gamma*r3*(g_best - x)

        global_coefficient/global_best are optional extensions beyond the
        minimal spec (which only requires personal-best + informants-best);
        left at 0 by default so behaviour matches the pseudocode exactly
        unless a caller opts in.
        """
        rng = rng or np.random.default_rng()
        r1 = rng.random(self.position.shape)
        r2 = rng.random(self.position.shape)

        cognitive = cognitive_coefficient * r1 * (self.best_position - self.position)

        if informants_best is not None:
            social = social_coefficient * r2 * (informants_best - self.position)
        else:
            social = 0.0

        global_term = 0.0
        if global_best is not None and global_coefficient:
            r3 = rng.random(self.position.shape)
            global_term = global_coefficient * r3 * (global_best - self.position)

        self.velocity = inertia_weight * self.velocity + cognitive + social + global_term

    def update_position(self, step_size: float = 1.0) -> None:
        """x(t+1) = x(t) + epsilon * v(t+1)."""
        self.position = self.position + step_size * self.velocity
