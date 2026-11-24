"""Population: manages particles and runs the main PSO loop (section 3.6/3.7)."""

from typing import List, Tuple

import numpy as np

from src.pso.fitness_function import FitnessFunction
from src.pso.informant_strategy import InformantSelectionStrategy
from src.pso.particle import Particle


class Population:
    """Owns all particles, the fitness function and the informant strategy;
    implements optimise() per the pseudocode in section 3.7.
    """

    def __init__(
        self,
        fitness_function: FitnessFunction,
        informant_strategy: InformantSelectionStrategy,
        num_particles: int,
        num_informants: int,
        num_iterations: int,
        dimension: int,
        inertia_weight: float = 0.7,
        cognitive_coefficient: float = 1.5,
        social_coefficient: float = 1.5,
        global_coefficient: float = 0.0,
        step_size: float = 1.0,
        position_bounds: Tuple[float, float] = (-1.0, 1.0),
        velocity_bounds: Tuple[float, float] = (-0.1, 0.1),
        seed: int = None,
    ):
        if not (1 <= num_informants <= num_particles - 1):
            raise ValueError("num_informants k must satisfy 1 <= k <= N-1.")

        self.fitness_function = fitness_function
        self.informant_strategy = informant_strategy
        self.num_particles = num_particles
        self.num_informants = num_informants
        self.num_iterations = num_iterations
        self.dimension = dimension
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.global_coefficient = global_coefficient
        self.step_size = step_size
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds
        self.rng = np.random.default_rng(seed)

        self.particles: List[Particle] = []
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.fitness_history: List[float] = []  # best fitness per iteration, for convergence plots

    def initialise(self) -> None:
        low, high = self.position_bounds
        vlow, vhigh = self.velocity_bounds
        self.particles = []
        for _ in range(self.num_particles):
            position = self.rng.uniform(low, high, size=self.dimension)
            velocity = self.rng.uniform(vlow, vhigh, size=self.dimension)
            self.particles.append(Particle(position, velocity))
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.fitness_history = []

    def evaluate_particles(self) -> None:
        for particle in self.particles:
            particle.evaluate(self.fitness_function)

    def update_personal_bests(self) -> None:
        for particle in self.particles:
            particle.update_personal_best()

    def update_global_best(self) -> None:
        for particle in self.particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = particle.best_position.copy()

    def assign_informants(self) -> None:
        for particle in self.particles:
            particle.informants = self.informant_strategy.select_informants(
                particle, self.particles, self.num_informants
            )

    def optimise(self) -> Tuple[np.ndarray, float]:
        """Run the full PSO loop; return (global_best_position, global_best_fitness)."""
        self.initialise()
        self.evaluate_particles()
        self.update_personal_bests()
        self.update_global_best()
        self.fitness_history.append(self.global_best_fitness)

        for _ in range(self.num_iterations):
            self.assign_informants()
            for particle in self.particles:
                informants_best, _ = particle.get_informants_best()
                particle.update_velocity(
                    informants_best=informants_best,
                    global_best=self.global_best_position,
                    inertia_weight=self.inertia_weight,
                    cognitive_coefficient=self.cognitive_coefficient,
                    social_coefficient=self.social_coefficient,
                    global_coefficient=self.global_coefficient,
                    rng=self.rng,
                )
                particle.update_position(step_size=self.step_size)

            self.evaluate_particles()
            self.update_personal_bests()
            self.update_global_best()
            self.fitness_history.append(self.global_best_fitness)

        return self.get_best_solution()

    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        return self.global_best_position, self.global_best_fitness
