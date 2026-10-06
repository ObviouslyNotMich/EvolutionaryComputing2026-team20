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

Team changes (see "TEAM CHANGES" comments): distance alone rarely produced
grounded walking, so the fitness evolved into

    fitness = final XY distance
              + 1 * fraction of samples the core touches the terrain (dragging)
              + 5 * fraction of samples nothing touches the terrain (jumping)
              + 1 * mean shortfall below 2 cm core clearance above the feet

There is no height or airtime bonus. Lower is better.

HOW TO RUN
-------------------------------------

    A2=assignments/assignment_2/mich/A2_template_2026.py
    GAIT=assignments/assignment_2/mich/gait_search.py

1. ES only, random initialisation (one or n step sizes):

    .venv/bin/python $A2 --mutation one --seed 42 --workers 15 --output __data__/es_one_seed42
    .venv/bin/python $A2 --mutation n   --seed 42 --workers 15 --output __data__/es_n_seed42

2. Optional Optuna warm-up (needs optuna 5.x), then freeze its gait_best.npz:

    .venv/bin/python $GAIT --fixed-clock-hz 1 --trials 500 --workers 15 --seed 43 \
        --output __data__/gait_seed43

3. ES from that gait:

    for m in one n; do
      .venv/bin/python $A2 --mutation $m --seed 42 --workers 15 \
          --warm-start __data__/gait_seed43/gait_best.npz --output __data__/warm_${m}_seed42
    done

   Or steps 2+3 in one go (one mutation mode):

    .venv/bin/python $GAIT --fixed-clock-hz 1 --trials 500 --workers 15 --seed 43 \
        --train-after --mutation-mode one --output __data__/gait_then_es_seed43

4. Watch a saved controller (launcher = interactive, video = mp4, simple = headless):

    .venv/bin/python $A2 --replay __data__/warm_one_seed42/best.npz --mode launcher

Output folders must be new; each gets settings.json, fitness.csv (per generation),
best.npz (best-ever controller) and the ARIEL database. Defaults: mu=75,
lambda=525, 57 generations = 30,000 evaluations. For a quick smoke test add
--population 4 --generations 2. Repeat runs over several seeds for the report.
"""

# Standard library
from pathlib import Path
from typing import Literal
import argparse
import csv
import json
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
from ariel.utils.noise_gen import PerlinNoise

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
TARGET_POSITION: list[float] = [5.5, 0.0, 0.6]  # where it should end up (Olympic arena finish)
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "simple"  # see run_experiment() for the options
GENERATIONS = 100
TARGET_SIZE = 75 # Population size
OFFSPRING_SIZE = TARGET_SIZE * 7 # 1/7 ratio is recommended or 1/4 ratio.

WEIGHTS_SCALE = 0.5

type StepsizeType = Literal["n", "one"]
SIGMA_MODE: StepsizeType = "n"
SIGMA_INIT = WEIGHTS_SCALE * 0.2 # 20% of initial scale
SIGMA_BOUNDARY = SIGMA_INIT * 0.1 

# TEAM CHANGES: walking-aware fitness (see module docstring).
FITNESS_VERSION = "distance_contact_clearance_v2"
CORE_CONTACT_WEIGHT = 1.0
AIRBORNE_WEIGHT = 5.0
CLEARANCE_WEIGHT = 1.0
CLEARANCE_TARGET = 0.02  # metres between core underside and supporting feet
CONTACT_SETTLING_TIME = 1.0  # ignore the first second while the robot lands
SAMPLE_INTERVAL = 0.1

# TEAM CHANGES: optional Optuna warm start (gait_search.py). One exact gait,
# then gait + N(0, INIT_NOISE) for 80% of parents, 20% random init.
INIT_NOISE = 0.02
RANDOM_FRACTION = 0.2


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
class SeededOlympicArena(OlympicArena):
    """TEAM CHANGE: ARIEL's arena with fixed terrain for every worker/replay.

    ARIEL's PerlinNoise is otherwise unseeded, giving each process another world.
    """

    def __init__(self, *args, terrain_seed: int = 0, **kwargs) -> None:
        self.terrain_seed = terrain_seed
        super().__init__(*args, **kwargs)

    def _generate_heightmap(self) -> np.ndarray:
        # Same generation/masking as ARIEL; only the noise seed is fixed.
        size = self.rugged_resolution
        noise = PerlinNoise(seed=self.terrain_seed).as_grid(
            size, size, scale=self.rugged_hillyness, normalize=False,
        )
        u = np.linspace(0.0, 1.0, size)
        v = np.linspace(0.0, 1.0, size)
        U, V = np.meshgrid(u, v, indexing="xy")
        distance = np.minimum.reduce([U, 1.0 - U, V, 1.0 - V])
        t = np.clip(distance / getattr(self, "edge_width", 0.1), 0.1, 1.0)
        mask = t * t * (3.0 - 2.0 * t)
        return noise * mask


def build_world() -> OlympicArena:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    # TEAM CHANGE: same seeded rugged Olympic arena for every run.
    world = SeededOlympicArena(load_precompiled=False, terrain_seed=0)

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
    # TEAM CHANGE: XY only (20 inputs), matching the planar fitness.
    distance_at_spawn = np.linalg.norm(np.subtract(TARGET_POSITION[:2], SPAWN_POS[:2]))
    # Distance from current position to the target, scaled so it starts at 1 (to avoid saturation region of tanh).
    distance_to_target = (np.asarray(TARGET_POSITION[:2]) - data.qpos[0:2]) / distance_at_spawn
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

    TEAM CHANGE: planar distance plus dragging, flight and low-core penalties.
    """
    distance = distance_to_target(final_position, np.asarray(TARGET_POSITION))
    return (distance + CORE_CONTACT_WEIGHT * core_contact_fraction
            + AIRBORNE_WEIGHT * airborne_fraction + CLEARANCE_WEIGHT * low_core_shortfall)


class EvaluationTracker:
    """TEAM CHANGE: samples terrain contacts and core clearance every 0.1 s."""

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
        self.samples = self.core_contacts = self.airborne = self.upright = 0
        self.shortfalls: list[float] = []
        self.clearances: list[float] = []

    def core_underside(self, data: mj.MjData) -> float:
        rotation = data.geom_xmat[self.core].reshape(3, 3)
        return float((self.core_vertices @ rotation[2]).min() + data.geom_xpos[self.core][2])

    def contact_state(self, contacts) -> tuple[bool, bool, list[float]]:
        core_contact, leg_support, foot_heights = False, False, []
        for contact in contacts:
            if contact.dist > 0 or contact.efc_address < 0:  # active contacts only
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            # Ignore self-collisions and terrain--terrain contacts.
            if (a in self.robot_geoms) == (b in self.robot_geoms):
                continue
            if (a if a in self.robot_geoms else b) == self.core:
                core_contact = True
            else:
                leg_support = True  # Any non-core robot geom belongs to a leg.
                foot_heights.append(float(contact.pos[2]))
        return core_contact, leg_support, foot_heights

    def sample(self, data: mj.MjData) -> None:
        if data.time < self.next_sample:
            return
        self.next_sample += SAMPLE_INTERVAL
        core_contact, leg_support, feet = self.contact_state(data.contact[:data.ncon])
        self.samples += 1
        up = 1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)  # core z-axis vertical component
        self.upright += int(up > 0.5 and data.qpos[2] > 0.05)
        self.core_contacts += int(core_contact)
        self.airborne += int(not (core_contact or leg_support))
        if core_contact or leg_support:  # Flight is penalised separately.
            # Height above the feet works on rugged terrain; dragging = 0.
            clearance = 0.0 if core_contact or not feet else max(
                0.0, self.core_underside(data) - float(np.mean(feet)))
            self.clearances.append(clearance)
            self.shortfalls.append(max(0.0, CLEARANCE_TARGET - clearance) / CLEARANCE_TARGET)

    def summary(self, initial: npt.NDArray[np.float64], data: mj.MjData) -> dict:
        final = get_core_position(data)
        core = self.core_contacts / self.samples if self.samples else 1.0
        airborne = self.airborne / self.samples if self.samples else 1.0
        shortfall = float(np.mean(self.shortfalls)) if self.shortfalls else 1.0
        return {
            "fitness": fitness_function(initial, final, core, airborne, shortfall),
            "distance": distance_to_target(final, np.asarray(TARGET_POSITION)),
            "core_contact_fraction": core, "airborne_fraction": airborne,
            "low_core_shortfall": shortfall,
            "mean_clearance": float(np.mean(self.clearances)) if self.clearances else 0.0,
            "upright_fraction": self.upright / self.samples if self.samples else 0.0,
            "final_position": final.tolist(),
            "final_up": float(1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)),
            "final_z": float(data.qpos[2]),
        }


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #


def build_simulation() -> tuple[mj.MjModel, mj.MjData]:
    """Build, spawn and compile the world (shared by every evaluation path)."""
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
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
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    return model, data


def run_experiment(weights, mode: ViewerTypes = MODE, metrics: dict | None = None) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # --- World and robot --------------------------------------------------- #
    # TEAM CHANGE: training workers compile once and reuse (see initialise_worker).
    if mode == "simple" and _WORKER_MODEL is not None:
        model, data = _WORKER_MODEL, _WORKER_DATA
        mj.set_mjcb_control(None)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
    else:
        model, data = build_simulation()

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(controller_inputs(data))
    output_size = model.nu

    # weights = make_random_weights(input_size, output_size)
    tracker = EvaluationTracker(model)  # TEAM CHANGE

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
    # TEAM CHANGE: exploding simulations get a terrible score instead of NaN.
    unstable = any(data.warning[int(w)].number for w in (
        mj.mjtWarning.mjWARN_BADQPOS, mj.mjtWarning.mjWARN_BADQVEL,
        mj.mjtWarning.mjWARN_BADQACC, mj.mjtWarning.mjWARN_BADCTRL,
    ))
    if unstable or not np.isfinite(data.qpos).all():
        return 1e6
    summary = tracker.summary(initial_position, data)
    if metrics is not None:
        metrics.update(summary)
    fitness = summary["fitness"]

    if metrics is None:  # interactive/replay run, not an EA worker
        console.log(f"start  : {np.round(initial_position, 3)}")
        console.log(f"end    : {np.round(summary['final_position'], 3)}")
        console.log(f"target : {np.round(TARGET_POSITION, 3)}")
        console.log(f"distance {summary['distance']:.4f}; core contact "
                    f"{summary['core_contact_fraction']:.1%}; airborne "
                    f"{summary['airborne_fraction']:.1%}; clearance "
                    f"{summary['mean_clearance'] * 100:.1f} cm")
        console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


# TEAM CHANGE: process-parallel evaluation. Each worker compiles the world once.
_WORKER_MODEL: mj.MjModel | None = None
_WORKER_DATA: mj.MjData | None = None


def initialise_worker(clock_hz: float = 1.0) -> None:
    global _WORKER_MODEL, _WORKER_DATA, CLOCK_HZ
    CLOCK_HZ = clock_hz
    _WORKER_MODEL, _WORKER_DATA = build_simulation()


def evaluate_candidate(weights: list[npt.NDArray[np.float64]]) -> dict:
    metrics: dict = {}
    metrics["fitness"] = run_experiment(weights, mode="simple", metrics=metrics)
    return metrics


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

    # TEAM CHANGES: budget overrides for smoke tests, parallel workers,
    # output folder, optional Optuna warm start and checkpoint replay.
    parser.add_argument("--population", type=int, default=TARGET_SIZE, help="mu (lambda = 7 mu)")
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    parser.add_argument("--output", type=Path, help="New folder for database, CSV and best.npz")
    parser.add_argument("--warm-start", type=Path, metavar="GAIT_NPZ",
                        help="Optuna gait checkpoint from gait_search.py")
    parser.add_argument("--replay", type=Path, metavar="BEST_NPZ", help="Show a saved controller")
    parser.add_argument("--mode", choices=["launcher", "simple", "video", "frame", "no_control"],
                        default="launcher", help="Replay viewer mode")

    args = parser.parse_args()
    if args.population < 2 or args.generations < 1 or args.workers < 1:
        parser.error("Need population >= 2, generations >= 1 and workers >= 1")
    return args

def main() -> None:
    """Run the ES (optionally warm-started), or replay a saved controller."""

    args = parse_args()

    # Reassign seed with given cmd seed
    global RNG, TARGET_SIZE, OFFSPRING_SIZE, GENERATIONS
    RNG = np.random.default_rng(args.seed)
    set_seed(args.seed)
    TARGET_SIZE, GENERATIONS = args.population, args.generations
    OFFSPRING_SIZE = TARGET_SIZE * 7

    # TEAM CHANGE: warm-start/replay checkpoints must match this 1 Hz controller.
    checkpoint = args.replay or args.warm_start
    initial_weights = None
    if checkpoint:
        with np.load(checkpoint, allow_pickle=False) as saved:
            initial_weights = saved["weights"].copy()
            if "clock_hz" in saved and float(saved["clock_hz"]) != CLOCK_HZ:
                raise SystemExit(f"{checkpoint} was tuned at another clock frequency")

    # A quick look at the size of the problem you are about to search.
    model, data = build_simulation()

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
    if initial_weights is not None and initial_weights.shape != (num_weights,):
        raise SystemExit("Saved controller does not match this robot/network")

    if args.replay:
        split = input_size * HIDDEN_SIZE
        run_experiment([initial_weights[:split].reshape(input_size, HIDDEN_SIZE),
                        initial_weights[split:].reshape(HIDDEN_SIZE, output_size)],
                       mode=args.mode)
        return

    # Standard name of db using mutation stepsize mode and seed.
    db_name = f"db_{args.mutation}_{args.seed}"

    # TEAM CHANGE: one fresh folder per run; refuse to overwrite results.
    output = args.output or DATA / f"es_{args.mutation}_{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps({
        "seed": args.seed, "mutation": args.mutation, "mu": TARGET_SIZE,
        "lambda": OFFSPRING_SIZE, "generations": GENERATIONS,
        "evaluations": TARGET_SIZE + GENERATIONS * OFFSPRING_SIZE,
        "sigma_init": SIGMA_INIT, "sigma_boundary": SIGMA_BOUNDARY,
        "weights_scale": WEIGHTS_SCALE, "warm_start": str(args.warm_start) if args.warm_start else None,
        "init_noise": INIT_NOISE, "random_fraction": RANDOM_FRACTION,
        "world": "OlympicArena", "terrain_seed": 0, "robot": "spider_8",
        "spawn": SPAWN_POS, "target": TARGET_POSITION, "duration": SIM_DURATION,
        "clock_hz": CLOCK_HZ, "hidden_size": HIDDEN_SIZE, "fitness": FITNESS_VERSION,
        "core_contact_weight": CORE_CONTACT_WEIGHT, "airborne_weight": AIRBORNE_WEIGHT,
        "clearance_weight": CLEARANCE_WEIGHT, "clearance_target": CLEARANCE_TARGET,
        "settling_time": CONTACT_SETTLING_TIME, "sample_interval": SAMPLE_INTERVAL,
        "workers": args.workers,
    }, indent=2) + "\n")

    # Avoid BLAS thread oversubscription inside each simulation process.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialise_worker, initargs=(CLOCK_HZ,),
    ) as pool:
        ea = EvolutionStategies(input_size, output_size, args.mutation, db_name,
                                output_folder=output,
                                evaluate_batch=lambda batch: pool.map(evaluate_candidate, batch),
                                initial_weights=initial_weights)

        best_ind = ea.evolve()

    console.log("--- Results ---")
    console.log(f"best = {best_ind.fitness:.4f}; replay with --replay {output / 'best.npz'}")


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
                 output_folder: Path = Path("__data__"), evaluate_batch=None,
                 initial_weights: npt.NDArray[np.float64] | None = None) -> None:

        self.input_size = input_size
        self.output_size = output_size
        self.stepsize_type = stepsize_type

        self.config = EASettings(
            is_maximisation=False,  # minimize distance
            num_steps=GENERATIONS,
            target_population_size=TARGET_SIZE,
            output_folder=output_folder,
            db_file_name=db_file_name
        )

        # TEAM CHANGES: parallel evaluation, optional warm start, logging.
        self.evaluate_batch = evaluate_batch or (lambda batch: map(evaluate_candidate, batch))
        self.initial_weights = initial_weights
        self.generation = 0
        self.evaluations = 0
        self.best_fitness = float("inf")

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
        # TEAM CHANGE: identity membership, so value-equal clones cannot exceed mu.
        selected = {id(ind) for ind in best}
        for ind in population:
            if id(ind) not in selected:
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

        # Calculate fitness for all individuals
        # TEAM CHANGE: in parallel, keeping the fitness components as tags.
        decoded = [self.decode_weights(ind.genotype.get("weights")) for ind in to_eval]
        for ind, metrics in zip(to_eval, self.evaluate_batch(decoded), strict=True):
            ind.fitness = metrics["fitness"]
            ind.tags = metrics
            ind.requires_eval = False
            self.evaluations += 1

        return population

    def record(self, population: Population) -> Population:
        """TEAM CHANGE: log each generation and checkpoint the best-ever controller."""
        alive = population.alive.sort(sort="min")
        scores = np.array([ind.fitness for ind in alive])
        best = alive[0]
        tags = best.tags or {}
        folder = self.config.output_folder
        if best.fitness < self.best_fitness:
            self.best_fitness = best.fitness
            temporary = folder / "best.tmp.npz"
            np.savez(
                temporary, weights=best.genotype["weights"], sigma=best.genotype["stepsizes"],
                input_size=self.input_size, output_size=self.output_size,
                hidden_size=HIDDEN_SIZE, clock_hz=CLOCK_HZ, fitness=best.fitness,
                generation=self.generation, evaluations=self.evaluations,
                mutation_mode=self.stepsize_type, fitness_version=FITNESS_VERSION,
                **{key: tags.get(key, np.nan) for key in (
                    "distance", "core_contact_fraction", "airborne_fraction",
                    "low_core_shortfall", "mean_clearance")},
            )
            temporary.replace(folder / "best.npz")
        csv_path = folder / "fitness.csv"
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(["generation", "evaluations", "best", "mean", "worst",
                                 "best_so_far", "distance", "core_contact_fraction",
                                 "airborne_fraction", "mean_clearance"])
            writer.writerow([self.generation, self.evaluations, scores.min(), scores.mean(),
                             scores.max(), self.best_fitness, tags.get("distance"),
                             tags.get("core_contact_fraction"), tags.get("airborne_fraction"),
                             tags.get("mean_clearance")])
        console.log(f"gen {self.generation}/{GENERATIONS} evals {self.evaluations}: "
                    f"best={scores.min():.4f} mean={scores.mean():.4f} "
                    f"best_so_far={self.best_fitness:.4f} distance={tags.get('distance', np.nan):.4f} "
                    f"core={tags.get('core_contact_fraction', np.nan):.0%} "
                    f"air={tags.get('airborne_fraction', np.nan):.0%} "
                    f"clearance={tags.get('mean_clearance', np.nan) * 100:.1f}cm")
        self.generation += 1
        return population

    def evolve(self) -> Individual | None:
        """Runs the evolutaion strategies algorithm"""

        console.log("Evolving")
        # Make population
        individuals = [
            self.make_individual() for _ in range(self.config.target_population_size)
        ]

        # TEAM CHANGE: optional Optuna warm start. One exact gait, noisy gait
        # copies for 80% of parents, and 20% keep random initialisation.
        if self.initial_weights is not None:
            seeded = max(1, int(len(individuals) * (1 - RANDOM_FRACTION)))
            for index, ind in enumerate(individuals[:seeded]):
                weights = np.array(self.initial_weights, dtype=float)
                if index:
                    weights += RNG.normal(scale=INIT_NOISE, size=weights.size)
                ind.genotype = {"weights": weights.tolist(),
                                "stepsizes": ind.genotype["stepsizes"]}

        population = Population(individuals)
        population = self.evaluate(population)
        self.record(population)  # TEAM CHANGE

        ops = [
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
            EAOperation(self.record),  # TEAM CHANGE: logging/checkpoint only
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
