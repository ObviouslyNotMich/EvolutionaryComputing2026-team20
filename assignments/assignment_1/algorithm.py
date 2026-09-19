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
    _prune_invalid_edges,
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

SEED = 147
RNG = np.random.default_rng(SEED)
random.seed(SEED)

NUM_MODULES = 20

GENERATIONS = 100
POP_SIZE = 75

TOURNAMENT_SIZE = 4
P_MUTATION = 0.1

class Assignment1EA:
    def __init__(self, targets) -> None:
        self.targets = targets
        self.config = EASettings(
            is_maximisation=False, # minimization
            num_steps=GENERATIONS,
            target_population_size=POP_SIZE,
            output_folder=Path("__data__"), db_file_name="database.db"
        )

    def make_individual(self) -> Individual:
        """TODO: implement individual generation."""

        ind = Individual()
        genome = random_tree(max_modules=NUM_MODULES)
        ind.genotype = genome.to_dict()

        ind.tags = {"ps" : 0}
        ind.requires_eval = True
        
        return ind

    
    def parent_selection_tournament(self, population: Population) -> Population:
        """
        Tournament Selection.
        tournament_size = 5 chosen because of example ea_ackley in docs
        """
        for ind in population:
            # clear last generation's parent flags 
            ind.tags = {"ps" : 0}

        # Only evaluated, alive individuals can become parents.
        candidates = [ind for ind in population.alive if not ind.requires_eval]

       # Run parent selection as many times as the target population
        num_parents = self.config.target_population_size

        for _ in range(num_parents):
            # Pick an amount of competitors, without replacement
            competitors = RNG.choice(candidates, size=TOURNAMENT_SIZE, replace=False)

            winner = min(competitors, key=lambda ind: ind.fitness)

            # Assign selection to winner.
            winner.tags["ps"] += 1
        
        return population
    

    def survivor_selection_tournament(self, population: Population) -> Population:

        num_alive = len(population.alive)

        while num_alive > self.config.target_population_size:
            candidates = [ind for ind in population.alive if not ind.requires_eval]

            # Make tournament
            # k = min(TOURNAMENT_SIZE, len(candidates))

            competitors = RNG.choice(candidates, size=TOURNAMENT_SIZE, replace=False)

            # Kill the individual with the highest fitness
            doomed = max(competitors, key=lambda ind: ind.fitness)

            doomed.alive = False
            num_alive -= 1

        return population


    def mutation(self, genome: TreeGenome) -> TreeGenome:

        # Keeps retrying the mutation, untill it is valid (does not exceed max modules)
        while(True):
            new = copy.deepcopy(genome)

            # Swap with random subtree.
            mutate_subtree_replacement(new, max_modules=NUM_MODULES)
            _prune_invalid_edges(new)

            if len(new.nodes) <= NUM_MODULES:
                validate_genome_dict(new.to_dict())
                return new
            else:
                continue

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[TreeGenome, TreeGenome]:

        g1 = TreeGenome.from_dict(parent1.genotype)
        g2 = TreeGenome.from_dict(parent2.genotype)

        g1_child, g2_child = crossover_subtree(g1,g2)

        return g1_child, g2_child


    def reproduction(self, population: Population) -> Population:
        # Get all the parents selected for reproduction
        parents = [ind for ind in population if ind.tags.get("ps", 0) > 0]

        # print(len(parents))

        offspring: list[Individual] = []

        # Grow relative to who is actually alive now, not a fixed target,
        # so the population can expand generation over generation.
        
        target_pool = self.config.target_population_size * 2

        # Compute weigths for tournament wins, more wins have higher probability
        weights = np.array([ind.tags.get("ps", 0) for ind in parents], dtype=float)
        weights /= weights.sum()
        
        while len(population) + len(offspring) < target_pool:
            # Always do crossover
            # if P_CROSSOVER > RNG.random():
            # p1, p2 = random.sample(parents, 2) # Take two parents randomly
            # Select parents

            # Choose parents with probability weights
            p1, p2 = RNG.choice(parents, size=2, replace=False, p=weights)

            c1, c2 = self.crossover(p1, p2)
            
            # Make children
            child1 = Individual()
            child1.genotype = c1.to_dict()
            child1.tags["ps"] = 0 # no parent selection
            offspring.append(child1)
            
            child2 = Individual()
            child2.genotype = c2.to_dict()
            child2.tags["ps"] = 0 # no parent selection
            offspring.append(child2)
            
        for ind in offspring:
            if P_MUTATION > RNG.random():
                g_mutated = self.mutation(TreeGenome.from_dict(ind.genotype))

                ind.genotype = g_mutated.to_dict()
                
        # Add offspring to population
        population.extend(offspring)
        
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
            if ind.alive and ind.requires_eval
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
            num_steps=self.config.num_steps,
            is_maximisation=self.config.is_maximisation,
        )
        ea.run()

        return ea.get_solution("best", only_alive=False)

    # def random_search(self, population: Population) -> Population:
    #     """Random search: generate new individuals and evaluate them."""
    #     # Generate new individuals
    #     new_individuals = [
    #         self.make_individual() for _ in range(self.config.target_population_size)
    #     ]
        
    #     new_population = Population(new_individuals)
        

    #     return new_population
    
    
    # def random_evolve(self) -> Individual | None:
    #     """Run the evolutionary algorithm with random search."""
    #     population = Population([
    #         self.make_individual() for _ in range(self.config.target_population_size)
    #     ])

    #     # initial eval
    #     population = self.evaluate(population)

    #     ops = [
    #         EAOperation(self.evaluate),
    #         EAOperation(self.random_search),
    #     ]
        
    #     ea = EA(
    #         population,
    #         operations=ops,
    #         num_steps=STEPS,
    #         is_maximisation=self.config.is_maximisation,
    #     )
    #     ea.run()

    #     return ea.get_solution("best", only_alive=False)