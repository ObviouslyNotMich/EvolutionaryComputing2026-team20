from ariel.ec import (
    EA,
    EAOperation,
    Individual,
    Population,
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
    """TODO: implement crossover. """

    return population

@EAOperation
def mutate(population: Population) -> Population:
    """TODO: implement mutate. """

    return population

@EAOperation
def survivor_selection(population: Population) -> Population:
    """"TODO: implement survivor selection"""
    
    return population
