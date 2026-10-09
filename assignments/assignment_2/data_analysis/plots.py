from pathlib import Path
from typing import Literal
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


#
# Functions for plotting
#

def plot_fitness(databases, labels=None, title="Fitness per generation", minimum=True, maximum=True, mean=True):
    """
    Plot fitness curves from multiple SQLite databases.

    Parameters
    ----------
    databases : list[str | Path]
        Paths to the .db files.
    labels : list[str] | None
        Optional labels. Defaults to the file names.
    title : str
        Plot title.
    """
    
    databases = [Path(database) for database in databases]
    
    if labels is None:
        labels = [database.stem for database in databases]
    
    if len(databases) != len(labels):
        raise ValueError("The number of labels must match the number of databases.")
    
    plt.figure(figsize=(10, 5.5))
    
    for database, label in zip(databases, labels):
        mean_fitness = get_mean_fitness(database)
        max_fitness = get_max_fitness(database)
        min_fitness = get_min_fitness(database)
        
        if mean_fitness.empty:
            print(f"Skipping {database.name}: no evaluated individuals found.")
            continue
        
        if mean:
            plt.plot(
                mean_fitness["generation"],
                mean_fitness["mean_fitness"],
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=f"Mean: {label}",
            )

        if minimum:
            plt.plot(
                min_fitness["generation"],
                min_fitness["min_fitness"],
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=f"Min: {label}",
            )

        if maximum:
            plt.plot(
                max_fitness["generation"],
                max_fitness["max_fitness"],
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=f"Max: {label}",
            )
    
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Experiment")
    plt.tight_layout()
    plt.show()


def get_min_fitness(db_path):
    """Load the minimum recorded fitness for each generation."""
    
    query = """
        SELECT
            time_of_birth AS generation,
            fitness_
        FROM individual
        WHERE fitness_ IS NOT NULL
    """
    
    with sqlite3.connect(db_path) as connection:
        data = pd.read_sql_query(query, connection)
    
    if data.empty:
        return pd.DataFrame(columns=["generation", "min_fitness"])
    
    return (
        data.groupby("generation", as_index=False)["fitness_"]
        .min()
        .rename(columns={"fitness_": "min_fitness"})
    )


def get_max_fitness(db_path):
    """Load the maximum recorded fitness for each generation."""
    
    query = """
        SELECT
            time_of_birth AS generation,
            fitness_
        FROM individual
        WHERE fitness_ IS NOT NULL
    """
    
    with sqlite3.connect(db_path) as connection:
        data = pd.read_sql_query(query, connection)
    
    if data.empty:
        return pd.DataFrame(columns=["generation", "max_fitness"])
    
    return (
        data.groupby("generation", as_index=False)["fitness_"]
        .max()
        .rename(columns={"fitness_": "max_fitness"})
    )


def get_mean_fitness(db_path):
    """Load and aggregate mean fitness per generation from one SQLite database."""
    
    query = """
        SELECT
            time_of_birth AS generation,
            fitness_
        FROM individual
        WHERE fitness_ IS NOT NULL
    """
    
    with sqlite3.connect(db_path) as connection:
        data = pd.read_sql_query(query, connection)
    
    if data.empty:
        return pd.DataFrame(columns=["generation", "mean_fitness"])
    
    return (
        data.groupby("generation", as_index=False)["fitness_"]
        .mean()
        .rename(columns={"fitness_": "mean_fitness"})
    )