"""Plot mean +- std over seeds of per-generation fitness from ea_test.py databases.

    uv run assignments/assignment_2/ai_gen_plot_ea_test.py
Reads   __data__/ea_test/<mode>_seed*/database.db
Writes  __data__/ea_test/fitness.png
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path.cwd() / "__data__" / "ea_test"
MODES = {"single": "[W1,W2,W3]·σ", "layer": "Wi·σi", "random": "random search"}


def run_curves(db: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per generation: best-so-far and mean fitness of the individuals born that generation."""
    rows = sqlite3.connect(db).execute(
        "SELECT time_of_birth, MIN(fitness_), AVG(fitness_) FROM individual "
        "WHERE fitness_ IS NOT NULL GROUP BY time_of_birth ORDER BY time_of_birth"
    ).fetchall()
    _, best, mean = map(np.array, zip(*rows))
    return np.minimum.accumulate(best), mean


def main() -> None:
    fig, (ax_best, ax_mean) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for mode, label in MODES.items():
        runs = [run_curves(db) for db in sorted(DATA.glob(f"{mode}_seed*/database.db"))]
        if not runs:
            continue
        n = min(len(r[0]) for r in runs)  # truncate to shortest run
        for ax, k in ((ax_best, 0), (ax_mean, 1)):
            curves = np.array([r[k][:n] for r in runs])
            m, s = curves.mean(0), curves.std(0)
            ax.plot(m, label=f"{label} (n={len(runs)})")
            ax.fill_between(range(n), m - s, m + s, alpha=0.25)

    ax_best.set(title="Best so far", xlabel="generation", ylabel="distance to target (lower = better)")
    ax_mean.set(title="Generation mean (offspring)", xlabel="generation")
    ax_best.legend()
    fig.tight_layout()
    fig.savefig(DATA / "fitness.png", dpi=150)
    print(f"saved {DATA / 'fitness.png'}")


if __name__ == "__main__":
    main()
