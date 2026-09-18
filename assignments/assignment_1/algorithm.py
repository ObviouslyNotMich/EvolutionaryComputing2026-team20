from ariel.ec import (
    EA,
    EAOperation,
    EASettings,
    Individual,
    Population,
)

from pathlib import Path


from ariel.ec.genotypes.tree.tree_genome import TreeGenome

import random

from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_subtree_replacement,
    random_tree,
)


# Local scripts
from tree_edit_distance import (
    mean_plus_std_tree_edit_distance,
)

import networkx as nx
import copy
import numpy as np
from ariel.ec.genotypes.tree.operators import random_tree, mutate_subtree_replacement, crossover_subtree
from ariel.ec.genotypes.tree.validation import validate_genome_dict

NUM_OF_MODULES: int = 20  # module budget per evolved body

SEED = 42
RNG = np.random.default_rng(SEED)

STEPS = 75
NUM_MODULES = 20
P_MUTATION = 0.05
POP_SIZE = 100
TOURNAMENT_SIZE = 4
NUM_ELITES = 1

class Assignment1EA:
    def __init__(self, targets) -> None:
        self.targets = targets
        self.config = EASettings(
            is_maximisation=False, # minimization
            num_steps=STEPS,
            target_population_size=POP_SIZE,
            output_folder=Path("__data__"), db_file_name="database.db"
        )

    def make_individual(self) -> Individual:
        """TODO: implement individual generation."""

        ind = Individual()
        genome = random_tree(max_modules=NUM_MODULES)
        ind.genotype = genome.to_dict()

        ind.tags["ps"] = False
        ind.tags["valid"] = True
        
        return ind
    
    def parent_selection(self, population: Population) -> Population:
        
        for ind in population:
            # clear last generation's parent flags 
            # because tags persist accross generations
            ind.tags = {"ps" : False}
        
        population = population.sort(sort="min", attribute="fitness_") #get top 50%
        cutoff = len(population) // 2

        # Give ps tag to all 'selected' individuals
        for i, ind in enumerate(population):
            ind.tags["ps"] = i < cutoff

        return population
    
    def parent_selection_tournament(self, population: Population) -> Population:
        """
        Tournament Selection.
        tournament_size = 5 chosen because of example ea_ackley in docs
        """
        for ind in population:
            # clear last generation's parent flags 
            # because tags persist accross generations
            ind.tags = {"ps" : False}

        # Only evaluated individuals can become parents
        candidates = [ind for ind in population.alive if ind.fitness_ is not None]

        # We need at least two parents for a child
        if len(candidates) < 2:
            return population

        # Two parents per child, one child per population slot
        num_parents = 2 * self.config.target_population_size

        for _ in range(num_parents):
            competitors = [random.choice(candidates) for _ in range(TOURNAMENT_SIZE)]

            winner = min(competitors, key=lambda ind: ind.fitness)

            winner.tags = {
                "ps": True,
            }
        
        return population
    

    def survivor_selection(self, population: Population) -> Population:
        
        population = population.sort(sort="min", attribute="fitness_") #get top 50%
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False

        return population

    def survivor_selection_tournament(self, population: Population) -> Population:

        for ind in population.alive:
            if ind.fitness_ is None:
                ind.alive = False

        alive = population.alive
        
        ranked = alive.sort(sort="min", attribute="fitness_")
        
        
        elite_ids = {id(ind) for ind in ranked[:NUM_ELITES]}  # Keep the best individuals alive

        num_alive = len(alive)
        while num_alive > self.config.target_population_size:
            candidates = [ind for ind in population.alive if id(ind) not in elite_ids]
            if not candidates:
                break

            k = min(TOURNAMENT_SIZE, len(candidates))
            competitors = [random.choice(candidates) for _ in range(k)]

            doomed = min(competitors, key=lambda ind: ind.fitness)

            doomed.alive = False
            num_alive -= 1

        return population

    # ----------------------------------------------------------------

    def mutation(self, genome: TreeGenome) -> TreeGenome:

        new = copy.deepcopy(genome)

        # Swap with random subtree.
        mutate_subtree_replacement(new, max_modules=NUM_MODULES)
        validate_genome_dict(new.to_dict())

        return new


    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[TreeGenome, TreeGenome]:

        g1 = TreeGenome.from_dict(parent1.genotype)
        g2 = TreeGenome.from_dict(parent2.genotype)

        g1_child, g2_child = crossover_subtree(g1,g2)

        return g1_child, g2_child


    def reproduction(self, population: Population) -> Population:
        # Get all the parents selected for reproduction
        parents = [ind for ind in population if ind.tags.get("ps", False)]

        offspring: list[Individual] = []

        # Define a new pool with twice the size of the target
        target_pool = self.config.target_population_size * 2

        while len(population) + len(offspring) < target_pool:
            # Always do crossover
            # if P_CROSSOVER > RNG.random():
            p1, p2 = random.sample(parents, 2) # Take two parents randomly
            c1, c2 = self.crossover(p1, p2)
            
            # Make children
            child1 = Individual()
            child1.genotype = c1.to_dict()
            child1.tags["ps"] = False # no parent selection
            child1.tags["valid"] = True
            offspring.append(child1)
            
            child2 = Individual()
            child2.genotype = c2.to_dict()
            child2.tags["ps"] = False # no parent selection
            child2.tags["valid"] = True
            offspring.append(child2)
            
        # Add offspring to population
        population.extend(offspring)
            
        for ind in population:
            if P_MUTATION > RNG.random():
                g_mutated = self.mutation(TreeGenome.from_dict(ind.genotype))

                ind.genotype = g_mutated.to_dict()

        return population

    def fitness_function(self,
        body: nx.DiGraph,
        targets: list[nx.DiGraph],
    ) -> float:
        """Score one body against the whole target set. LOWER IS BETTER.

        Some things worth thinking about:
        * The std term charges for unevenness - body that is mediocre against every target
            and one that is excellent on most but bad on one can still land close
            in fitness, but the latter is penalized a bit more.
        * Nothing here rewards small bodies. Does your EA bloat? Should a size
            penalty be part of fitness, or is that the encoding's job?
        """
        return mean_plus_std_tree_edit_distance(body, targets)


    def evaluate(self, population: Population) -> Population:
        to_eval = [
            ind
            for ind in population
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]

        if not to_eval:
            return population

        for ind in to_eval:
            genome = TreeGenome.from_dict(ind.genotype)
            graph_genome = genome.to_networkx()

            fitness = self.fitness_function(graph_genome, self.targets)
            ind.fitness = fitness
            ind.requires_eval = False

        return population


    def evolve(self) -> Individual | None:
        """Run the evolutionary algorithm."""
        population = Population([
            self.make_individual() for _ in range(self.config.target_population_size)
        ])

        # initial eval
        population = self.evaluate(population)

        ops = [
            EAOperation(self.parent_selection_tournament),
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection_tournament),
        ]
        
        ea = EA(
            population,
            operations=ops,
            num_steps=STEPS,
            is_maximisation=self.config.is_maximisation,
        )
        ea.run()

        return ea.get_solution("best", only_alive=False)
