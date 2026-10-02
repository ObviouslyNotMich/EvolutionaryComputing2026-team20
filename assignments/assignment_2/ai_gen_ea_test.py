"""Self-adaptive (mu, lambda)-ES for neuroevolution on the OlympicArena.

Research question: one step size shared by all weight layers
    [W1, W2, W3] * sigma
vs one step size per layer
    W1 * sigma1, W2 * sigma2, W3 * sigma3
Baseline: random search with the same evaluation budget (mu + gens * lambda).

Run:
    uv run assignments/assignment_2/ai_gen_ea_test.py --mode single --seed 0
    uv run assignments/assignment_2/ai_gen_ea_test.py --mode layer  --seed 0
    uv run assignments/assignment_2/ai_gen_ea_test.py --mode random --seed 0
Results: __data__/ea_test/<mode>_seed<seed>/database.db
Plot:    uv run assignments/assignment_2/ai_gen_plot_ea_test.py
"""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import mujoco as mj
import numpy as np

from ariel import console
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import spider_8
from ariel.ec import EA, EAOperation, Individual, Population, config, set_seed
import ariel.simulation.environments.olympic_arena as olympic_arena
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner

# --- EXPERIMENT CONSTANTS (fixed across all runs) --- #
SPAWN_POS = [-0.8, 0.0, 0.1]
TARGET_POSITION = [2.0, 0.0, 0.1]
SIM_DURATION = 15.0
HIDDEN_SIZE = 8
CPG_FREQ = 1.0  # Hz of the sin/cos clock input
SIGMA_INIT = 0.1
SIGMA_MIN = 1e-3  # epsilon_0, Eiben & Smith 4.4.2
FAIL_FITNESS = 1e3  # sim blew up (NaN)
TERRAIN_SEED = 2026

# OlympicArena builds its rugged section with unseeded Perlin noise -> a new map
# every evaluation. Inject a fixed seed (runtime patch, ariel source untouched).
olympic_arena.PerlinNoise = partial(olympic_arena.PerlinNoise, seed=TERRAIN_SEED)

RNG = np.random.default_rng()  # reseeded in main()


# ============================================================================ #
#  WORLD / BODY / CONTROLLER
# ============================================================================ #
def build_model() -> tuple[mj.MjModel, mj.MjData]:
    mj.set_mjcb_control(None)  # DO NOT REMOVE
    world = OlympicArena()
    world.spawn(spider_8().spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    return model, data


def layer_shapes() -> list[tuple[int, int]]:
    model, data = build_model()
    n_in = len(data.qpos) + 4  # qpos + (dx, dy) to target + sin/cos clock
    return [(n_in, HIDDEN_SIZE), (HIDDEN_SIZE, HIDDEN_SIZE), (HIDDEN_SIZE, model.nu)]


def controller_inputs(data: mj.MjData) -> np.ndarray:
    to_target = np.asarray(TARGET_POSITION[:2]) - data.qpos[:2]
    phase = 2 * np.pi * CPG_FREQ * data.time
    return np.concatenate([data.qpos, to_target, [np.sin(phase), np.cos(phase)]])


def nn_controller(data: mj.MjData, weights: list[np.ndarray]) -> np.ndarray:
    w1, w2, w3 = weights
    h = np.tanh(controller_inputs(data) @ w1)
    h = np.tanh(h @ w2)
    return np.tanh(h @ w3) * (np.pi / 2)


def fitness_function(final_position: np.ndarray) -> float:
    """XY distance to target. LOWER IS BETTER."""
    return float(np.linalg.norm(final_position[:2] - np.asarray(TARGET_POSITION[:2])))


def simulate(weights_lists: list) -> float:
    """One headless evaluation. Module-level so multiprocessing can pickle it."""
    weights = [np.asarray(w) for w in weights_lists]
    model, data = build_model()

    def cb(m: mj.MjModel, d: mj.MjData) -> None:
        d.ctrl[:] = nn_controller(d, weights)

    mj.set_mjcb_control(cb)
    simple_runner(model, data, duration=SIM_DURATION)
    mj.set_mjcb_control(None)

    fit = fitness_function(data.qpos[:3])
    return fit if np.isfinite(fit) else FAIL_FITNESS


# ============================================================================ #
#  ES OPERATORS
# ============================================================================ #
# Genotype: {"w": [W1, W2, W3] (nested lists), "sigma": [s] or [s1, s2, s3]}
ARGS: argparse.Namespace  # set in main()
SHAPES: list[tuple[int, int]]
POOL = None  # multiprocessing.Pool, set in main()


def make_individual() -> Individual:
    ind = Individual()
    n_sigma = 1 if ARGS.mode != "layer" else len(SHAPES)
    ind.genotype = {
        "w": [RNG.normal(scale=0.5, size=s).tolist() for s in SHAPES],
        "sigma": [SIGMA_INIT] * n_sigma,
    }
    return ind


def recombine(p1: dict, p2: dict) -> dict:
    """Discrete on weights, intermediary on sigmas (Eiben & Smith 6.5)."""
    w = []
    for a, b in zip(p1["w"], p2["w"]):
        a, b = np.asarray(a), np.asarray(b)
        w.append(np.where(RNG.random(a.shape) < 0.5, a, b))
    sigma = ((np.asarray(p1["sigma"]) + np.asarray(p2["sigma"])) / 2).tolist()
    return {"w": w, "sigma": sigma}


def mutate(geno: dict) -> dict:
    """Self-adaptive Gaussian mutation, sigma first, then weights."""
    n = sum(a * b for a, b in SHAPES)
    sigma = np.asarray(geno["sigma"])
    if len(sigma) == 1:  # one step size: sigma' = sigma * exp(tau0 * N)
        sigma = sigma * np.exp(RNG.normal() / np.sqrt(n))
    else:  # per-layer: sigma_i' = sigma_i * exp(tau' * N + tau * N_i)
        tau_global, tau_local = 1 / np.sqrt(2 * n), 1 / np.sqrt(2 * np.sqrt(n))
        sigma = sigma * np.exp(tau_global * RNG.normal() + tau_local * RNG.normal(size=len(sigma)))
    sigma = np.maximum(sigma, SIGMA_MIN)

    layer_sigmas = np.broadcast_to(sigma, len(SHAPES))
    w = [(np.asarray(wl) + s * RNG.normal(size=np.shape(wl))).tolist()
         for wl, s in zip(geno["w"], layer_sigmas)]
    return {"w": w, "sigma": sigma.tolist()}


def reproduce(population: Population) -> Population:
    """Uniform random parent selection -> recombination -> mutation, lambda times.
    In random mode: lambda fresh random individuals (random-search baseline)."""
    parents = population.alive.to_list()
    for _ in range(ARGS.lam):
        if ARGS.mode == "random":
            population.append(make_individual())
            continue
        i, j = RNG.choice(len(parents), size=2, replace=False)
        child = Individual()
        child.genotype = mutate(recombine(parents[i].genotype, parents[j].genotype))
        population.append(child)
    return population


def evaluate(population: Population) -> Population:
    todo = population.unevaluated.to_list()
    genos = [ind.genotype["w"] for ind in todo]
    fits = POOL.map(simulate, genos) if POOL else map(simulate, genos)
    for ind, fit in zip(todo, fits):
        ind.fitness = fit
    return population


def survivor_selection(population: Population) -> Population:
    """(mu, lambda): parents die, best mu offspring survive."""
    offspring = [ind for ind in population if ind.time_of_birth == -1]  # not yet committed
    for ind in population:
        if ind.time_of_birth != -1:
            ind.alive = False
    offspring.sort(key=lambda ind: ind.fitness)
    for ind in offspring[ARGS.mu:]:
        ind.alive = False

    fits = [ind.fitness for ind in offspring[: ARGS.mu]]
    sig = np.mean([np.mean(ind.genotype["sigma"]) for ind in offspring[: ARGS.mu]])
    console.log(f"best {fits[0]:.3f}  mean {np.mean(fits):.3f}  mean sigma {sig:.4f}")
    return population


# ============================================================================ #
#  MAIN
# ============================================================================ #
def main() -> None:
    global ARGS, SHAPES, POOL, RNG
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "layer", "random"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--mu", type=int, default=10)
    p.add_argument("--lam", type=int, default=70)  # mu/lambda ~ 1/7
    p.add_argument("--workers", type=int, default=4)
    ARGS = p.parse_args()

    RNG = np.random.default_rng(ARGS.seed)
    set_seed(ARGS.seed)
    SHAPES = layer_shapes()
    console.log(f"layers {SHAPES}, genotype length {sum(a * b for a, b in SHAPES)}")

    out = Path.cwd() / "__data__" / "ea_test" / f"{ARGS.mode}_seed{ARGS.seed}"
    config.target_population_size = ARGS.mu
    config.is_maximisation = False

    POOL = Pool(ARGS.workers) if ARGS.workers > 1 else None
    try:
        population = evaluate(Population([make_individual() for _ in range(ARGS.mu)]))
        ops = [EAOperation(reproduce), EAOperation(evaluate), EAOperation(survivor_selection)]
        ea = EA(population, ops, num_steps=ARGS.gens, is_maximisation=False,
                db_file_path=out / "database.db")
        ea.run()
        best = ea.get_solution("best", only_alive=False)
        console.log(f"best fitness {best.fitness:.4f}, sigma {best.genotype['sigma']}")
    finally:
        if POOL:
            POOL.close()


if __name__ == "__main__":
    main()
