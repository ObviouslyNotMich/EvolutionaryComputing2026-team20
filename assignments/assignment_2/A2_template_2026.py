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
GENERATIONS = 30
TARGET_SIZE = 10 # Population size
OFFSPRING_SIZE = TARGET_SIZE * 7 # 1/7 ratio is recommended or 1/4 ratio.

WEIGHTS_SCALE = 0.5

type StepsizeType = Literal["n", "one"]
SIGMA_MODE: StepsizeType = "n"
SIGMA_INIT = WEIGHTS_SCALE * 0.2 # 20% of initial scale
SIGMA_BOUNDARY = SIGMA_INIT * 0.1 


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
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    The plain version: how far is the robot from the target when time runs out?

    `initial_position` is unused here on purpose - it is passed in because the
    moment you want a less naive fitness you will need it. Some things worth
    thinking about (and, ideally, comparing in your report):
      * Distance *reduced* rather than distance remaining, so a robot that
        starts closer is not rewarded for standing still.
      * Penalising a robot that falls over or leaves the arena.
      * Whether the z-axis should count at all - a robot that jumps is not
        closer to the target in any way you care about.
    See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
    """
    target = np.asarray(TARGET_POSITION)
    return float(np.linalg.norm(final_position[:2] - target[:2]))


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #


def run_experiment(weights, mode: ViewerTypes = MODE) -> float:
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

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

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

    # fitness = fitness_function(initial_position, final_position)
    fitness = distance_to_target(final_position, TARGET_POSITION)

    # console.log(f"start  : {np.round(initial_position, 3)}")
    # console.log(f"end    : {np.round(final_position, 3)}")
    # console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    # console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""
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

    ea = EvolutionStategies(input_size, output_size, SIGMA_MODE)

    best_ind = ea.evolve()

    console.log("--- Results ---")
    console.log(f"best = {best_ind}")

    weights = best_ind.genotype.get("weights")
    decoded_weights = ea.decode_weights(weights)
    # print(decoded_weights)
    run_experiment(decoded_weights, mode="launcher")


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
    def __init__(self, input_size: int, output_size: int, stepsize_type: StepsizeType) -> None:

        self.input_size = input_size

        self.output_size = output_size

        self.stepsize_type = stepsize_type

        self.config = EASettings(

            is_maximisation=False,  # minimize distance probably

            num_steps=GENERATIONS,

            target_population_size=TARGET_SIZE,

            output_folder=Path("__data__"), db_file_name="database.db"
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
            "stepsizes": [mut_stepsize],
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

        # Calculate fitness for all individuals
        for ind in to_eval:
            weights = ind.genotype.get("weights")
            decoded_weights = self.decode_weights(weights)
            # print(decoded_weights)
            ind.fitness = run_experiment(decoded_weights, mode="simple")
            ind.requires_eval = False

        best = population.best(sort="min")
        for ind in best:
            print(f"Best fitness: {ind.fitness}")

        return population

    def evolve(self) -> Individual | None:
        """Runs the evolutaion strategies algorithm"""

        console.log("Evolving")
        # Make population
        population = Population([
            self.make_individual() for _ in range(self.config.target_population_size)
        ])

        console.log("Initial population created")
        population = self.evaluate(population)
        console.log("Initial population evaluated")

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
        )

        ea.run()

        return ea.get_solution("best", only_alive=False)


if __name__ == "__main__":
    main()
