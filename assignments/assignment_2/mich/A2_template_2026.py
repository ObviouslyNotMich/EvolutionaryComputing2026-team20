"""EC A2 template code - neuroevolution for targeted locomotion with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo*, not a solution. It spawns a robot, drives it with a neural network
whose weights are RANDOM, runs the simulation, and reports how close the robot
ended up to a target.

There is deliberately NO evolution in here. Building the EA (representation,
initialisation, parent selection, variation, survivor selection) is the assignment.
See "YOUR JOB" at the bottom of this file.

THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve the weights of a neural network controller so that a robot moves from
SPAWN_POS to TARGET_POSITION within the simulation time.

    fitness = distance between the robot's final position and TARGET_POSITION

HOW TO RUN
----------

Change MODE below to switch between an interactive viewer, a headless run,
a rendered video, or a single frame.
"""

# Standard library
from pathlib import Path
from typing import Literal
import argparse
import json
import sqlite3
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import spider_8, snake, spider_12
from ariel.ec import set_seed, EA, EASettings, EAOperation, Individual, Population
from ariel.simulation.environments import SimpleFlatWorld, OlympicArena, CraterTerrainWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.tasks.targeted_locomotion import fitness_delta_distance, distance_to_target

# Type aliases
type ViewerTypes = Literal["launcher",
                           "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = 42
RNG = np.random.default_rng(SEED)

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
TARGET_POSITION: list[float] = [5.0 ,5.0, 0.1] # [5.5, 0.0, 0.6]  # where it should end up
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "simple"  # see run_experiment() for the options

# TODO Determine algorithm parameters
GENERATIONS = 2
TARGET_SIZE = 5 # Population size
OFFSPRING_SIZE = TARGET_SIZE * 7 # 1/7 ratio is recommended or 1/4 ratio.

WEIGHTS_SCALE = 0.5

type StepsizeType = Literal["n", "one"]
SIGMA_MODE: StepsizeType = "n"
SIGMA_INIT = WEIGHTS_SCALE * 0.2 # 20% of initial scale
SIGMA_BOUNDARY = SIGMA_INIT * 0.1 

# Fitness choice (--fitness): "distance" = original final XY distance only,
# "walking" = distance + penalties below.
type FitnessMode = Literal["distance", "walking"]
FITNESS_MODE: FitnessMode = "walking"

# Walking-aware fitness:
#   fitness = XY distance + 1 * core-contact fraction (dragging)
#             + 5 * airborne fraction (jumping) + 1 * low-core shortfall
# Contacts are sampled every 0.1 s after 1 s settling. No height/airtime bonus.
CORE_CONTACT_WEIGHT = 1.0
AIRBORNE_WEIGHT = 5.0
CLEARANCE_WEIGHT = 1.0
CLEARANCE_TARGET = 0.02  # metres between core underside and supporting feet
CONTACT_SETTLING_TIME = 1.0
SAMPLE_INTERVAL = 0.1


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    world = SimpleFlatWorld()

    return world


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return spider_8()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

# Controller architecture - decide before writing your EA.
HIDDEN_SIZE: int = 6
CLOCK_HZ: float = 1.0 # Frequency of the sine/cosine clock inputs


def controller_inputs(data:mj.MjData) -> npt.NDArray[np.float64]:
    """The robot state + direction to the target + a clock + a constant bias"""
    # Straight line distance to the target at spawn 5.5m in our case
    distance_at_spawn = np.linalg.norm(np.subtract(TARGET_POSITION[:3], SPAWN_POS[:3]))
    # Distance from current position to the target, scaled so it starts at 1 (to avoid saturation region of tanh).
    distance_to_target = (np.asarray(TARGET_POSITION[:3]) - data.qpos[0:3]) / distance_at_spawn
    phase = 2 * np.pi * CLOCK_HZ * data.time
    clock = [np.sin(phase), np.cos(phase)]
    return np.concatenate([data.qpos, distance_to_target, clock, [1.0]])


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below.
    inputs = controller_inputs(data)

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
    ]


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
#
# The robot is spawned with a free joint, so data.qpos[0:3] IS the core's
# (x, y, z) world position. Read it before and after stepping - no tracker or
# bookkeeping needed. (`data.geom("robot1_core").xpos` works too.)
#
# ============================================================================ #


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
    core_contact_fraction: float = 0.0,
    airborne_fraction: float = 0.0,
    low_core_shortfall: float = 0.0,
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    Planar distance plus dragging, flight and low-core penalties.
    """
    distance = distance_to_target(final_position, np.asarray(TARGET_POSITION))
    return (distance + CORE_CONTACT_WEIGHT * core_contact_fraction
            + AIRBORNE_WEIGHT * airborne_fraction + CLEARANCE_WEIGHT * low_core_shortfall)


class EvaluationTracker:
    """Samples terrain contacts and core clearance every SAMPLE_INTERVAL seconds."""

    def __init__(self, model: mj.MjModel) -> None:
        self.core = model.geom("robot1_core").id
        self.robot_geoms = {
            i for i in range(model.ngeom)
            if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or "").startswith("robot1_")
        }
        # Core mesh vertices in the geom frame, for the true (tilted) underside.
        mesh = model.geom_dataid[self.core]
        start = model.mesh_vertadr[mesh]
        self.core_vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].copy()
        self.next_sample = CONTACT_SETTLING_TIME
        self.samples = self.core_contacts = self.airborne = 0
        self.shortfalls: list[float] = []

    def core_underside(self, data: mj.MjData) -> float:
        rotation = data.geom_xmat[self.core].reshape(3, 3)
        return float((self.core_vertices @ rotation[2]).min() + data.geom_xpos[self.core][2])

    def sample(self, data: mj.MjData) -> None:
        if data.time < self.next_sample:
            return
        self.next_sample += SAMPLE_INTERVAL
        core_contact, feet = False, []
        for contact in data.contact[:data.ncon]:
            if contact.dist > 0 or contact.efc_address < 0:  # active contacts only
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.robot_geoms) == (b in self.robot_geoms):  # self/terrain-only
                continue
            if (a if a in self.robot_geoms else b) == self.core:
                core_contact = True
            else:
                feet.append(float(contact.pos[2]))  # any non-core robot geom is a leg
        self.samples += 1
        self.core_contacts += int(core_contact)
        if not (core_contact or feet):
            self.airborne += 1  # flight: penalised separately, no clearance credit
            return
        clearance = 0.0 if core_contact else max(0.0, self.core_underside(data) - np.mean(feet))
        self.shortfalls.append(max(0.0, CLEARANCE_TARGET - clearance) / CLEARANCE_TARGET)

    def metrics(self, initial: npt.NDArray[np.float64], final: npt.NDArray[np.float64]) -> dict:
        """All fitness components, so runs with either fitness can be compared."""
        n = max(1, self.samples)
        core, airborne = self.core_contacts / n, self.airborne / n
        shortfall = float(np.mean(self.shortfalls)) if self.shortfalls else 1.0
        return {
            "distance": distance_to_target(final, TARGET_POSITION),
            "core_contact_fraction": core, "airborne_fraction": airborne,
            "low_core_shortfall": shortfall,
            "walking_fitness": fitness_function(initial, final, core, airborne, shortfall),
        }


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #


def run_experiment(weights, mode: ViewerTypes = MODE, metrics: dict | None = None) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(controller_inputs(data))
    output_size = model.nu

    # weights = make_random_weights(input_size, output_size)
    tracker = EvaluationTracker(model)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions
        tracker.sample(d)

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            # Interactive window. Great for seeing what your robot does,
            # useless inside an evolutionary loop.
            viewer.launch(model=model, data=data)

            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "simple":
            # Headless. THIS is the one your EA uses.
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            # Render to an mp4 - for the figures in your report.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            # A single image of the scene. Useful to check your spawn position
            # and that the robot is not clipping through the floor.
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            # No controller attached: drag the hinges around by hand.
            viewer.launch(model=model, data=data)

    # Detach the callback again so the next run starts clean.
    mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    # console.log(f"Final position: {final_position}")

    if not np.isfinite(data.qpos).all():
        return 1e6
    components = tracker.metrics(initial_position, final_position)
    if metrics is not None:
        metrics.update(components)
    if FITNESS_MODE == "distance":
        # Original fitness: final XY distance only.
        fitness = distance_to_target(final_position, TARGET_POSITION)
    else:
        fitness = components["walking_fitness"]

    # console.log(f"start  : {np.round(initial_position, 3)}")
    # console.log(f"end    : {np.round(final_position, 3)}")
    # console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    # console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def initialise_worker(fitness_mode: FitnessMode) -> None:
    """Worker processes re-import this file, so pass the fitness choice on."""
    global FITNESS_MODE
    FITNESS_MODE = fitness_mode


def evaluate_candidate(weights: list[npt.NDArray[np.float64]]) -> tuple[float, dict]:
    """One headless evaluation; runs in a worker process."""
    metrics: dict = {}
    return run_experiment(weights, mode="simple", metrics=metrics), metrics


def compare(db_files: list[Path]) -> None:
    """Re-simulate each run's best controller and print the SAME measures for all.

    Fitness values of the two modes are not comparable with each other; raw
    distance, core contact, airborne time and clearance are.
    """
    mj.set_mjcb_control(None)
    world = build_world()
    world.spawn(build_robot().spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    input_size, output_size = len(controller_inputs(data)), model.nu
    split = input_size * HIDDEN_SIZE
    console.log(f"{'run':<28} {'fitness':>9} {'distance':>9} {'core':>6} {'air':>6} "
                f"{'low-core':>9} {'walking':>9}")
    for db_file in db_files:
        with sqlite3.connect(db_file) as db:
            genotype, fitness = db.execute(
                "SELECT genotype_, fitness_ FROM individual WHERE fitness_ IS NOT NULL "
                "ORDER BY fitness_ LIMIT 1").fetchone()
        flat = np.array(json.loads(genotype)["weights"] if isinstance(genotype, str)
                        else genotype["weights"])
        weights = [flat[:split].reshape(input_size, HIDDEN_SIZE),
                   flat[split:].reshape(HIDDEN_SIZE, output_size)]
        m: dict = {}
        run_experiment(weights, mode="simple", metrics=m)
        console.log(f"{Path(db_file).name:<28} {fitness:9.4f} {m['distance']:9.4f} "
                    f"{m['core_contact_fraction']:6.1%} {m['airborne_fraction']:6.1%} "
                    f"{m['low_core_shortfall']:9.1%} {m['walking_fitness']:9.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EA2026 Assignment 2"
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=SEED,
        help="Seed used for random functions",
    )

    parser.add_argument(
        "--mutation",
        "-m",
        choices=["n", "one"],
        default=SIGMA_MODE,
        help=f"Mutation step-size mode. Either 'n' or 'one' Default: {SIGMA_MODE}",
    )

    parser.add_argument(
        "--fitness",
        "-f",
        choices=["distance", "walking"],
        default=FITNESS_MODE,
        help=f"'distance' (original) or 'walking' (distance + penalties). Default: {FITNESS_MODE}",
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        metavar="DB",
        help="Compare the best controller of each database on the same measures",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel simulation processes",
    )

    return parser.parse_args()

def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""

    args = parse_args()
    if args.compare:
        compare(args.compare)
        return

    # Reassign seed with given cmd seed
    global RNG, FITNESS_MODE
    FITNESS_MODE = args.fitness
    RNG = np.random.default_rng(args.seed)
    set_seed(args.seed)

    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(controller_inputs(data))
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")
    console.log(f"mutation stepsize mode             : {args.mutation}")
    console.log(f"seed                               : {args.seed}")

    # Standard name of db using mutation stepsize mode and seed.
    db_name = f"db_{args.fitness}_{args.mutation}_{args.seed}"

    # Processes, not threads: MuJoCo's control callback is global per process.
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialise_worker, initargs=(args.fitness,),
    ) as pool:
        ea = EvolutionStategies(input_size, output_size, args.mutation, db_name,
                                evaluate_batch=lambda batch: pool.map(evaluate_candidate, batch))

        best_ind = ea.evolve()

    console.log("--- Results ---")
    console.log(f"best = {best_ind}")

    # weights = best_ind.genotype.get("weights")
    # decoded_weights = ea.decode_weights(weights)
    # # print(decoded_weights)
    # run_experiment(decoded_weights, mode="launcher")


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #


class EvolutionStategies:
    def __init__(self, input_size: int, output_size: int, stepsize_type: StepsizeType, db_file_name: str,
                 evaluate_batch=None) -> None:

        self.input_size = input_size
        self.output_size = output_size
        self.stepsize_type = stepsize_type
        self.evaluate_batch = evaluate_batch or (lambda batch: map(evaluate_candidate, batch))

        self.config = EASettings(
            is_maximisation=False,  # minimize distance
            num_steps=GENERATIONS,
            target_population_size=TARGET_SIZE,
            output_folder=Path("__data__"), 
            db_file_name=db_file_name
        )

    def make_individual(self) -> Individual:
        ind = Individual()

        # Single vector with all weights (NEEDS TO BE CONTSTRUCTED INTO MATRICES AGAIN AFTER)
        n = self.input_size * HIDDEN_SIZE + self.output_size * HIDDEN_SIZE

        match self.stepsize_type:
            case "n":
                n_sigma = n
            case "one":
                n_sigma = 1

        # Random weights and predefined sigma
        ind.genotype = {
            "weights": RNG.normal(scale=WEIGHTS_SCALE, size=n).tolist(),
            "stepsizes": np.full(n_sigma, SIGMA_INIT).tolist(),
        }

        ind.requires_eval = True
        return ind

    def mutation_onestep(self, ind: Individual) -> Individual:
        """Gaussian perturbation, used in reproduction()"""

        # Assume n_sigma = 1
        genome = ind.genotype

        # Get stepsize first
        weights = np.array(genome.get("weights"))
        stepsize = np.array(genome.get("stepsizes"))

        tau = 1 / np.sqrt(len(weights))
        global_noise = RNG.normal()

        # New mutation step size
        mut_stepsize = stepsize * np.exp(tau * global_noise)

        # Check stepsize does not exceed boundary
        mut_stepsize = np.maximum(mut_stepsize, SIGMA_BOUNDARY)

        weights = weights + mut_stepsize * RNG.normal(size=len(weights))

        ind.genotype = {
            "weights": weights.tolist(),
            "stepsizes": mut_stepsize.tolist(),
        }

        return ind

    def mutation_nstep(self, ind: Individual) -> Individual:
        """Gaussian perturbation, used in reproduction()"""

        genome = ind.genotype

        # Get stepsize first
        weights = np.array(genome.get("weights"))
        stepsizes = np.array(genome.get("stepsizes"))

        assert len(weights) == len(
            stepsizes), "Weights and stepsizes are not equal in length"

        # See 4.4.2 in Introduction to Evolutionary Computing
        tau = 1 / np.sqrt(2 * np.sqrt(len(weights)))
        tau_prime = 1 / np.sqrt(2 * len(weights))

        global_noise = RNG.normal()
        local_noise = RNG.normal(size=len(stepsizes))

        # Calculate new mutated stepsizes, check for boundary
        stepsizes = stepsizes * \
            np.exp(tau_prime * global_noise + tau * local_noise)
        stepsizes = np.maximum(stepsizes, SIGMA_BOUNDARY)

        # Mutate weights with new stepsizes
        weights = weights + stepsizes * RNG.normal(size=len(weights))

        # Assign back to individual
        ind.genotype = {
            "weights": weights.tolist(),
            "stepsizes": stepsizes.tolist(),
        }

        return ind

    def recombination(self, population: Population) -> Individual:
        """Discrete or intermediary, used in reproduction()"""
        # print(f"recombination start: population size = {len(population)}")

        # TODO Amount of parents TBD, research (Contemporary Evolution Strategies)
        # Number of parents used for recombination
        # num_parents = round(len(population) * 0.2)

        alive = population.alive

        # Uniform random parent selection
        # Choose all parents for global recombination (recommended)
        num_parents = max(2, len(alive))
        parents = RNG.choice(alive, size=num_parents, replace=False)

        # Empty child
        child = Individual()
        child.requires_eval = True

        # Get weigths and stepsizes from all parents
        parent_weights = [np.array(p.genotype["weights"]) for p in parents]
        parent_stepsizes = [np.array(p.genotype["stepsizes"]) for p in parents]

        # Discrete recombination on weights
        child_weights = [float(RNG.choice(col))
                         for col in zip(*parent_weights)]

        # Intermediate recombination on stepsizes
        child_stepsizes = [float(np.mean(col))
                           for col in zip(*parent_stepsizes)]

        child.genotype = {
            "weights": child_weights,
            "stepsizes": child_stepsizes,
        }

        return child

    def reproduction(self, population: Population) -> Population:
        # console.log("Reproducing...")

        offspring: list[Individual] = []

        # Make offspring with previous generation
        while len(offspring) < OFFSPRING_SIZE:
            offspring.append(self.recombination(population))

        # Choose mutation type
        match self.stepsize_type:
            case "n":
                mutate = self.mutation_nstep
            case "one":
                mutate = self.mutation_onestep

        # Mutate individuals
        for i, individual in enumerate(offspring):
            offspring[i] = mutate(individual)

        # Kill previous generation
        for individual in population:
            individual.alive = False

        # Add new offspring to population
        population.extend(offspring)
        return population

    def survivor_selection(self, population: Population) -> Population:
        """Deterministic elitist replacement by (mu, lambda), selects only from offspring"""

        # Only choose from alive population (new generation)
        alive = population.alive

        # (mu + lambda) is worse for self adaption according to theory
        best = alive.best(
            sort="min", n=self.config.target_population_size)

        # print(f"best count: {len(best)}")
        # print(f"fitnesses: {[ind.fitness for ind in population]}")

        # Kill rest of remaining population that are not scoring high enough
        for ind in population:
            if ind not in best:
                ind.alive = False

        return population

    def decode_weights(self, weights: npt.NDArray[np.float64]):

        weights = np.array(weights)

        split_point = self.input_size * HIDDEN_SIZE

        input_to_hidden = weights[:split_point].reshape(
            self.input_size, HIDDEN_SIZE)
        hidden_to_output = weights[split_point:].reshape(
            HIDDEN_SIZE, self.output_size)

        return [input_to_hidden, hidden_to_output]

    def evaluate(self, population: Population) -> Population:
        """Evaluation function, look at ariel.simulation.tasks.targeted_locomotion for inspiration"""
        # TODO General idea, make it walk to the finish in the olympic arena.

        to_eval = [
            ind for ind in population.alive if ind.requires_eval
        ]

        if not to_eval:
            return population

        # Calculate fitness for all individuals (in parallel)
        decoded = [self.decode_weights(ind.genotype.get("weights")) for ind in to_eval]
        for ind, (fitness, metrics) in zip(to_eval, self.evaluate_batch(decoded), strict=True):
            ind.fitness = fitness
            ind.tags = metrics  # all components stored in the database for comparison
            ind.requires_eval = False

        # best = population.best(sort="min")
        # for ind in best:
        #     print(f"Best fitness: {ind.fitness}")

        return population

    def evolve(self) -> Individual | None:
        """Runs the evolutaion strategies algorithm"""

        console.log("Evolving")
        # Make population
        population = Population([
            self.make_individual() for _ in range(self.config.target_population_size)
        ])
        population = self.evaluate(population)

        ops = [
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
        ]

        # Run algorithm for generations
        ea = EA(
            population,
            operations=ops,
            num_steps=self.config.num_steps,
            is_maximisation=self.config.is_maximisation,
            db_file_path=self.config.output_folder / self.config.db_file_name
        )

        ea.run()

        return ea.get_solution("best", only_alive=False)


if __name__ == "__main__":
    main()
