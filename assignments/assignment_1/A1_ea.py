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
def parent_selection(population: Population, tournament_size = 5) -> Population:
    """
    Tournament Selection.
    tournament_size = 5 chosen because of example ea_ackley in docs
    """
    for ind in population:
        # clear last generation's parent flags 
        # because tags persist accross generations
        ind.tags = {"ps" : False, "ps_count":0}

    # Only evaluated individuals can become parents
    candidates = [ind for ind in population.alive if ind.fitness_ is not None]

    # We need at least two parents for a child
    if len(candidates) < 2:
        return population

    # Two parents per child, one child per population slot
    num_parents = 2 * config.target_population_size

    for _ in range(num_parents):
        competitors = [random.choice(candidates) for _ in range(tournament_size)]

        winner = min(competitors, key=lambda ind: ind.fitness)

        winner.tags = {
            "ps": True,
            "ps_count": int(winner.tags.get("ps_count", 0)) + 1,
        }

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
def survivor_selection(
    population: Population,
    tournament_size: int = 5,
    num_elites: int = 1
    ) -> Population:

    for ind in population.alive:
        if ind.fitness_ is None:
            ind.alive = False

    alive = population.alive.to_list()
    ranked = sorted(
        alive,
        key=lambda ind: ind.fitness,
        reverse=config.is_maximisation,
    )
    elite_ids = {id(ind) for ind in ranked[:num_elites]}

    num_alive = len(alive)
    while num_alive > config.target_population_size:
        candidates = [ind for ind in population.alive if id(ind) not in elite_ids]
        if not candidates:
            break

        k = min(tournament_size, len(candidates))
        competitors = [random.choice(candidates) for _ in range(k)]
        if config.is_maximisation:
            doomed = min(competitors, key=lambda ind: ind.fitness)
        else:
            doomed = max(competitors, key=lambda ind: ind.fitness)

        doomed.alive = False
        num_alive -= 1

    return population




