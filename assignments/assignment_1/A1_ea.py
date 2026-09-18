from ariel.ec import (
    EA,
    N_OFFSPRING,
    MATING_POOL,
    EAOperation,
    Individual,
)

import numpy as np


def random_genotype(num_modules: int):
    """"TODO: implement random genotype generation."""

    return

@EAOperation
def make_individual() -> Individual:
    """TODO: implement individual generation."""
    ind = Individual()
    # Give genotype here....
    
    return ind

@EAOperation
def evaluate(population: Population) -> Population:
    """TODO: implement evaluation selection. """

    return population

@EAOperation
def parent_selection(population: Population) -> Population:
    """TODO: implement parent selection. """

    return population

@EAOperation
def crossover(population: Population) -> Population:
    """Make offspring by swapping subtrees between two parents.

    `crossover_subtree` picks a random not core node in each parent and swaps
    the whole branch hanging below it, giving two children. It deep copies
    first, so the parents are untouched, and if a swap produces an invalid
    body it returns copies of the parents instead. This way crossover sometimes does nothing.

    We draw both parents freshly at random each time rather than shuffling the
    pool once and pairing neighbours. Fixed pairing would mean parent 0 could
    only ever breed with parent 1, which explores far less. 
    And should help avoid the problem of a single very good parent dominating the population, 
    which can happen if it is paired with a weak neighbour.
    """
    if len(_MATING_POOL) < 2:
        return population
    for _ in range(N_OFFSPRING // 2):  # 2 children per crossover
        a, b = random.sample(MATING_POOL, 2)
        for kid in crossover_subtree(to_genome(a.genotype), to_genome(b.genotype)):
            child = Individual()
            child.genotype = cap_size(kid).to_dict()
            child.tags = {"mutate": True}  # flag it for the mutation step
            population.append(child)
    return population

@EAOperation
def mutate(population: Population, probability: float) -> Population:
    '''
    Selects a random non-core node, removes its entire subtree,
    and replaces it with a new randomly generated subtree.
    '''
    for i, genome in enumerate(population):
        if random.random() < probability:
            mutate_subtree_replacement(
                genome,
                max_modules=max_modules
            )
        validate_genome_dict(genome_sm.to_dict())
    return population

@EAOperation
def survivor_selection(population: Population) -> Population:
    """"TODO: implement survivor selection"""
    
    return population
