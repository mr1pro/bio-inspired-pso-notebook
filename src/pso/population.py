"""The swarm and the main PSO loop.

This file implements Algorithm 39 (Particle Swarm Optimization) from
S. Luke, Essentials of Metaheuristics, 2nd edition, Section 3.5. Tags of the
form "[A39 L12]" name the line of that algorithm a piece of code implements.
For reference, the algorithm is:

     1: swarmsize <- desired swarm size
     2: alpha   <- proportion of velocity to be retained
     3: beta    <- proportion of personal best to be retained
     4: gamma   <- proportion of the informants' best to be retained
     5: delta   <- proportion of global best to be retained
     6: epsilon <- jump size of a particle
     7: P <- {}
     8: for swarmsize times do
     9:     P <- P + {new random particle x with a random initial velocity v}
    10: Best <- empty
    11: repeat
    12:     for each particle x in P with velocity v do
    13:         AssessFitness(x)
    14:         if Best = empty or Fitness(x) > Fitness(Best) then
    15:             Best <- x
    16:     for each particle x in P with velocity v do      (determine how to mutate)
    17:         x* <- previous fittest location of x
    18:         x+ <- previous fittest location of informants of x (including x itself)
    19:         x! <- previous fittest location of any particle
    20:         for each dimension i do
    21:             b <- random number from 0.0 to beta inclusive
    22:             c <- random number from 0.0 to gamma inclusive
    23:             d <- random number from 0.0 to delta inclusive
    24:             v_i <- alpha*v_i + b(x*_i - x_i) + c(x+_i - x_i) + d(x!_i - x_i)
    25:     for each particle x in P with velocity v do      (mutate)
    26:         x <- x + epsilon*v
    27: until Best is the ideal solution or we have run out of time
    28: return Best

Things added on top of Algorithm 39 (all optional and off by default):
    * interchangeable informant selection (src/pso/informant_strategy.py)
    * boundary handling                   (src/pso/boundary.py)
    * a limit on the velocity             (max_velocity)
    * an inertia weight that decreases linearly during the run
"""

from typing import List, Optional, Tuple

import numpy as np

from src.pso.boundary import BoundaryHandler, NoBoundary
from src.pso.fitness_function import FitnessFunction
from src.pso.informant_strategy import InformantSelectionStrategy, RandomInformantSelection
from src.pso.particle import Particle


class Population:
    """Owns the particles and runs PSO.

    One iteration of the main loop evaluates every particle exactly once, so a
    run costs num_particles * num_iterations fitness evaluations.

    Args:
        fitness_function:  object with evaluate(position); higher is better.
        dimension:         length of the vectors being optimised.
        num_particles:     swarm size.                                  [A39 L1]
        num_iterations:    how many times the main loop runs ("time").  [A39 L27]
        inertia_weight:    alpha, share of the old velocity kept.       [A39 L2]
        cognitive_coefficient: beta, pull towards the personal best.    [A39 L3]
        social_coefficient:    gamma, pull towards the informants' best. [A39 L4]
        global_coefficient:    delta, pull towards the global best.     [A39 L5]
        step_size:         epsilon, jump size.                          [A39 L6]
        num_informants:    k, informants per particle (1 <= k <= N - 1).
        informant_strategy: how the k informants are chosen (default: random).
        init_bounds:       (low, high) range for the random start positions.
        boundary_handler:  what to do when a particle leaves the search space
                           (default: nothing, as in Algorithm 39).
        velocity_init:     "half_diff" (half the vector between two random
                           points, the method the book describes) or "zero".
        max_velocity:      if given, each velocity component is limited to
                           [-max_velocity, +max_velocity].
        final_inertia_weight: if given, alpha moves linearly from
                           inertia_weight to this value over the run.
        target_fitness:    stop early once this fitness is reached (the
                           "ideal solution" of line 27). None = never.
        track_diversity:   record the mean distance of the particles from the
                           swarm centre at every iteration.
        seed:              seed for all random numbers used by the run.
    """

    def __init__(
        self,
        fitness_function: FitnessFunction,
        dimension: int,
        num_particles: int = 30,
        num_iterations: int = 100,
        inertia_weight: float = 0.7,
        cognitive_coefficient: float = 1.5,
        social_coefficient: float = 1.5,
        global_coefficient: float = 0.0,
        step_size: float = 1.0,
        num_informants: int = 3,
        informant_strategy: Optional[InformantSelectionStrategy] = None,
        init_bounds: Tuple[float, float] = (-1.0, 1.0),
        boundary_handler: Optional[BoundaryHandler] = None,
        velocity_init: str = "half_diff",
        max_velocity: Optional[float] = None,
        final_inertia_weight: Optional[float] = None,
        target_fitness: Optional[float] = None,
        track_diversity: bool = False,
        seed: Optional[int] = None,
    ):
        if num_particles < 2:
            raise ValueError("A swarm needs at least 2 particles.")
        if not (1 <= num_informants <= num_particles - 1):
            raise ValueError("num_informants k must satisfy 1 <= k <= num_particles - 1.")
        if num_iterations < 1:
            raise ValueError("num_iterations must be at least 1.")
        if velocity_init not in ("half_diff", "zero"):
            raise ValueError("velocity_init must be 'half_diff' or 'zero'.")

        self.fitness_function = fitness_function
        self.dimension = int(dimension)
        self.num_particles = int(num_particles)            # [A39 L1] swarmsize
        self.num_iterations = int(num_iterations)
        self.inertia_weight = float(inertia_weight)        # [A39 L2] alpha
        self.cognitive_coefficient = float(cognitive_coefficient)  # [A39 L3] beta
        self.social_coefficient = float(social_coefficient)        # [A39 L4] gamma
        self.global_coefficient = float(global_coefficient)        # [A39 L5] delta
        self.step_size = float(step_size)                  # [A39 L6] epsilon
        self.num_informants = int(num_informants)
        self.init_bounds = (float(init_bounds[0]), float(init_bounds[1]))
        self.velocity_init = velocity_init
        self.max_velocity = max_velocity
        self.final_inertia_weight = final_inertia_weight
        self.target_fitness = target_fitness
        self.track_diversity = track_diversity

        # One generator drives everything random in a run (start positions,
        # the b/c/d numbers, informant choice, random boundary handling), so a
        # seed reproduces the run exactly.
        self.rng = np.random.default_rng(seed)
        self.informant_strategy = informant_strategy or RandomInformantSelection()
        self.informant_strategy.bind(self.rng)
        self.boundary_handler = boundary_handler or NoBoundary()
        self.boundary_handler.bind(self.rng)

        self.particles: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None   # "Best" / x!
        self.global_best_fitness = -np.inf
        self.evaluations = 0
        self.fitness_history: List[float] = []    # best fitness after each iteration
        self.diversity_history: List[float] = []  # swarm spread after each iteration

    # ------------------------------------------------------------------
    # Set-up
    # ------------------------------------------------------------------
    def initialise(self) -> None:
        """[A39 L7-10] Create the swarm and clear the record of the best."""
        low, high = self.init_bounds
        self.particles = []                                       # [A39 L7] P <- {}
        for _ in range(self.num_particles):                       # [A39 L8]
            position = self.rng.uniform(low, high, size=self.dimension)
            if self.velocity_init == "half_diff":
                # "choosing two random points in the space and using half the
                # vector from one to the other" (Luke, Section 3.5)
                other = self.rng.uniform(low, high, size=self.dimension)
                velocity = 0.5 * (other - position)
            else:
                velocity = np.zeros(self.dimension)
            self.particles.append(Particle(position, velocity))   # [A39 L9]

        self.global_best_position = None                          # [A39 L10] Best <- empty
        self.global_best_fitness = -np.inf
        self.evaluations = 0
        self.fitness_history = []
        self.diversity_history = []

    # ------------------------------------------------------------------
    # The steps of one iteration
    # ------------------------------------------------------------------
    def evaluate_particles(self) -> None:
        """[A39 L12-13] Assess the fitness of every particle."""
        for particle in self.particles:                           # [A39 L12]
            particle.evaluate(self.fitness_function)              # [A39 L13]
            self.evaluations += 1

    def update_personal_bests(self) -> None:
        """Keep each particle's x* up to date (read later at [A39 L17])."""
        for particle in self.particles:
            particle.update_personal_best()

    def update_global_best(self) -> None:
        """[A39 L14-15] Best <- x if x is fitter than Best."""
        for particle in self.particles:
            if self.global_best_position is None or particle.fitness > self.global_best_fitness:  # [A39 L14]
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()                               # [A39 L15]

    def assign_informants(self) -> None:
        """Choose each particle's informants for this iteration. They are
        needed for x+ at [A39 L18]. Re-drawing them every iteration is what
        the book describes for random informants."""
        self.informant_strategy.begin_iteration(self.particles)
        for particle in self.particles:
            particle.informants = self.informant_strategy.select_informants(
                particle, self.particles, self.num_informants
            )

    def current_inertia(self, iteration: int) -> float:
        """Alpha for this iteration: constant, or decreasing linearly from
        inertia_weight to final_inertia_weight if the latter is set."""
        if self.final_inertia_weight is None or self.num_iterations == 1:
            return self.inertia_weight
        progress = iteration / (self.num_iterations - 1)
        return self.inertia_weight + progress * (self.final_inertia_weight - self.inertia_weight)

    def swarm_diversity(self) -> float:
        """Mean Euclidean distance of the particles from their centre. Large
        values mean the swarm is spread out (exploring); values near zero mean
        it has collapsed onto one point."""
        positions = np.stack([p.position for p in self.particles])
        centre = positions.mean(axis=0)
        return float(np.mean(np.linalg.norm(positions - centre, axis=1)))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def optimise(self) -> Tuple[np.ndarray, float]:
        """Run PSO and return (best position, best fitness).  [A39 L7-28]"""
        self.initialise()                                         # [A39 L7-10]

        for iteration in range(self.num_iterations):              # [A39 L11] repeat
            self.evaluate_particles()                             # [A39 L12-13]
            self.update_personal_bests()                          # keeps x* for L17
            self.update_global_best()                             # [A39 L14-15]

            self.fitness_history.append(self.global_best_fitness)
            if self.track_diversity:
                self.diversity_history.append(self.swarm_diversity())

            # [A39 L27] "until Best is the ideal solution ..."
            if self.target_fitness is not None and self.global_best_fitness >= self.target_fitness:
                break

            alpha = self.current_inertia(iteration)
            self.assign_informants()

            # [A39 L16] Determine how to mutate: every velocity is updated
            # before any particle moves, exactly as in the algorithm.
            for particle in self.particles:
                # [A39 L17] x* is particle.best_position (used inside update_velocity)
                informants_best, _ = particle.get_informants_best()   # [A39 L18] x+
                global_best = self.global_best_position                # [A39 L19] x!
                particle.update_velocity(                              # [A39 L20-24]
                    informants_best=informants_best,
                    global_best=global_best,
                    alpha=alpha,
                    beta=self.cognitive_coefficient,
                    gamma=self.social_coefficient,
                    delta=self.global_coefficient,
                    rng=self.rng,
                    max_velocity=self.max_velocity,
                )

            for particle in self.particles:                       # [A39 L25] mutate
                particle.update_position(self.step_size)          # [A39 L26]
                # Extension: bring the particle back if it left the space.
                particle.position, particle.velocity = self.boundary_handler.apply(
                    particle.position, particle.velocity
                )
        # [A39 L27] "... or we have run out of time": the loop above ends
        # after num_iterations passes.

        return self.get_best_solution()                           # [A39 L28]

    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        """[A39 L28] return Best."""
        return self.global_best_position, self.global_best_fitness
