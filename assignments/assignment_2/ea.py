from pathlib import Path

from typing import Literal


# Third-party libraries

import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer


# Local libraries (ARIEL)

from ariel import console
from ariel.ec import (
    EA,
    EAOperation,
    EASettings,
    Individual,
    Population,
    set_seed
)


SEED = 42
RNG = np.random.default_rng(SEED)
set_seed(SEED)

# TODO Determine algorithm parameters
HIDDEN_SIZE: int = 6
GENERATIONS = 100
TARGET_SIZE = 75  # Population size
MUTATION_SIGMA = 0.1


class EvolutionStategies:
    def __init__(self, input_size: int, output_size: int) -> None:


        self.input_size = input_size

        self.output_size = output_size


        self.config = EASettings(

            is_maximisation=False,  # minimize distance probably

            num_steps=GENERATIONS,

            target_population_size=TARGET_SIZE,

            output_folder=Path("__data__"), db_file_name="database.db"
        )


        # TODO add target location for fitness????
        
    def make_individual(self) -> Individual:
        ind = Individual()

        # Make individual with random weights.
        # This depends on our NN architecture.
        # Also need to determine if self-adaptive

        ind.genotype = [
            RNG.normal(scale=0.5, size=(self.input_size, HIDDEN_SIZE)).tolist(),
            RNG.normal(scale=0.5, size=(HIDDEN_SIZE, self.output_size)).tolist()
        ]
        
        return ind


    def parent_selection(self, population: Population) -> Population:
        """Uniform random."""
        pass


    def survivor_selection(self, population: Population) -> Population:
        """Deterministic elitist replacement by (mu, lambda) or (mu + lambda)."""
        pass


    def mutation(self):
        """Gaussian perturbation, used in reproduction()"""
        pass


    def recombination(self):
        """Discrete or intermediary, used in reproduction()"""
        pass


    def evaluate(self):
        """Evaluation function, look at ariel.simulation.tasks.targeted_locomotion for inspiration"""
        pass


    def evolve(self) -> Individual | None:
        """Runs the evolutaion strategies algorithm"""

        # Make population
        population = Population([

            self.make_individual() for _ in range(self.config.target_population_size)

        ])

        # Evaluate initial population
        population = self.evaluate(population)

        ops = [

            EAOperation(self.parent_selection),
            EAOperation(self.mutation),
            EAOperation(self.recombination),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),

        ]

        # Run algorithm for generations
        ea = EA(

            population,

            operations=ops,

            num_steps=self.config.num_steps,

            is_maximisation=self.config.is_maximisation,

        )

        ea.run()

        return ea.get_solution("best", only_alive=False)
