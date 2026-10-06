import argparse
import csv
import json
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

import A2_template_2026 as demo
ROOT = Path(__file__).resolve().parent


def gait_weights(params: dict, input_size: int, output_size: int) -> np.ndarray:
    """Clock-only NN: tanh(sin), tanh(cos), constant, then per-hinge waves.

    The actuator ordering comes from the compiled model, not guessed leg order.
    ES can subsequently mutate all weights, including state/target feedback.
    """
    w1 = np.zeros((input_size, demo.HIDDEN_SIZE))
    w2 = np.zeros((demo.HIDDEN_SIZE, output_size))
    w1[-3, 0] = 1.0  # sine clock input
    w1[-2, 1] = 1.0  # cosine clock input
    w1[-1, 2] = 1.0  # constant input
    for hinge in range(output_size):
        amplitude = params[f"amplitude_{hinge}"]
        phase = params[f"phase_{hinge}"]
        w2[0, hinge] = amplitude * np.cos(phase)
        w2[1, hinge] = amplitude * np.sin(phase)
        w2[2, hinge] = params[f"bias_{hinge}"] / np.tanh(1.0)
    return np.concatenate([w1.ravel(), w2.ravel()])


def suggest_gait(trial, output_size, fixed_clock_hz=None):
    frequency = (trial.suggest_float("clock_hz", 0.4, 2.5, log=True)
                 if fixed_clock_hz is None
                 else trial.suggest_float("clock_hz", fixed_clock_hz, fixed_clock_hz))
    params = {"clock_hz": frequency}
    for hinge in range(output_size):
        params[f"amplitude_{hinge}"] = trial.suggest_float(f"amplitude_{hinge}", 0.05, 1.8)
        params[f"phase_{hinge}"] = trial.suggest_float(f"phase_{hinge}", -np.pi, np.pi)
        params[f"bias_{hinge}"] = trial.suggest_float(f"bias_{hinge}", -0.9, 0.9)
    return params


def split_weights(flat, input_size, output_size):
    split = input_size * demo.HIDDEN_SIZE
    return [flat[:split].reshape(input_size, demo.HIDDEN_SIZE),
            flat[split:].reshape(demo.HIDDEN_SIZE, output_size)]


def evaluate_gait(candidate):
    weights, frequency = candidate
    demo.CLOCK_HZ = frequency
    metrics = demo.evaluate_candidate(weights)
    score = metrics["fitness"]
    metrics.setdefault("upright_fraction", 0.0)
    metrics.setdefault("final_up", -1.0)
    metrics.setdefault("final_z", -1.0)
    return score, metrics


def gait_constraints(score, metrics):
    # Fitness is the template's walking-aware score; these are feasibility gates,
    # not a secretly different objective. Progress alone is not proof of walking.
    spawn_distance = np.linalg.norm(np.subtract(demo.TARGET_POSITION[:2], demo.SPAWN_POS[:2]))
    return [
        0.8 - metrics["upright_fraction"],
        0.5 - metrics["final_up"],
        0.05 - metrics["final_z"],
        metrics.get("distance", score) - (spawn_distance - 0.1),
    ]


def save_gait(folder, trial, input_size, output_size):
    temporary = folder / "gait_best.tmp.npz"
    np.savez(
        temporary, weights=gait_weights(trial.params, input_size, output_size),
        clock_hz=trial.params["clock_hz"], input_size=input_size,
        output_size=output_size, hidden_size=demo.HIDDEN_SIZE,
        fitness=trial.value, seed=trial.user_attrs["search_seed"],
        distance=trial.user_attrs["metrics"].get("distance", trial.value),
        fitness_version=demo.FITNESS_VERSION,
        **fitness_settings(),
    )
    temporary.replace(folder / "gait_best.npz")
    (folder / "gait_best.json").write_text(json.dumps({
        "trial": trial.number, "fitness": trial.value, "params": trial.params,
        "distance": trial.user_attrs["metrics"].get("distance", trial.value),
        "metrics": trial.user_attrs["metrics"],
        "constraints": trial.user_attrs["constraints"],
    }, indent=2) + "\n")


def fitness_settings():
    """The template's walking-aware fitness constants, recorded with every result."""
    return dict(core_contact_weight=demo.CORE_CONTACT_WEIGHT, airborne_weight=demo.AIRBORNE_WEIGHT,
                clearance_weight=demo.CLEARANCE_WEIGHT, clearance_target=demo.CLEARANCE_TARGET,
                settling_time=demo.CONTACT_SETTLING_TIME, sample_interval=demo.SAMPLE_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=43)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--output", type=Path)
    location.add_argument("--resume", type=Path, metavar="STUDY_FOLDER",
                          help="Continue the saved study; --trials is the total trial budget")
    parser.add_argument("--train-after", action="store_true")
    parser.add_argument("--fixed-clock-hz", type=float,
                        help="Fix rather than tune the clock; use 1 for paper experiments")
    parser.add_argument("--generations", type=int, default=demo.GENERATIONS)
    parser.add_argument("--mutation-mode", choices=["one", "n"], default="one")
    parser.add_argument("--population", type=int, default=75)
    parser.add_argument("--target-fitness", type=float, default=0.1)
    args = parser.parse_args()
    if (args.trials < 1 or args.workers < 1
            or args.generations < 1
            or args.population < 2 or not np.isfinite(args.target_fitness)
            or args.target_fitness < 0
            or (args.fixed_clock_hz is not None and (
                not np.isfinite(args.fixed_clock_hz) or args.fixed_clock_hz <= 0))):
        parser.error("Invalid search/training budget, clock or target")
    if args.train_after and args.fixed_clock_hz != 1.0:
        parser.error("Paper ES requires --fixed-clock-hz 1; tune a new study at 1 Hz")
    output = args.resume or args.output or Path("__data__") / f"a2_gait_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    if args.resume:
        if not (output / "optuna.db").is_file():
            parser.error("Resume folder must contain optuna.db")
        if args.train_after:
            parser.error("Resume is gait-only; inspect the gait before starting a new ES run")
    else:
        output.mkdir(parents=True, exist_ok=False)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    model, data = demo.build_simulation()
    input_size, output_size = len(demo.controller_inputs(data)), model.nu
    settings = vars(args).copy()
    settings["output"] = str(output)
    settings["resume"] = str(args.resume) if args.resume else None
    settings.update(
        world="OlympicArena", terrain_seed=0, robot="spider_8",
        duration=demo.SIM_DURATION, target=demo.TARGET_POSITION,
        objective=demo.FITNESS_VERSION, minimum_upright_fraction=0.8,
        **fitness_settings(),
        actuator_names=[model.actuator(i).name for i in range(output_size)],
        sigma_init=demo.SIGMA_INIT, init_noise=demo.INIT_NOISE, random_fraction=demo.RANDOM_FRACTION,
    )
    if args.resume:
        original = json.loads((output / "settings.json").read_text())
        for key in ("seed", "world", "terrain_seed", "robot", "duration", "target",
                    "objective", "minimum_upright_fraction", "actuator_names",
                    *fitness_settings()):
            if original.get(key) != settings[key]:
                parser.error(f"Resume would change {key}; create a new study instead")
        if original.get("fixed_clock_hz") != args.fixed_clock_hz:
            parser.error("Resume cannot change the frequency search; create a new study")
    else:
        (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=60, multivariate=True, constant_liar=True,
    )
    if args.resume:
        study = optuna.load_study(
            study_name="rhythmic_spider", storage=f"sqlite:///{output / 'optuna.db'}",
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            direction="minimize", study_name="rhythmic_spider",
            storage=f"sqlite:///{output / 'optuna.db'}", sampler=sampler,
        )
    existing = study.get_trials(deepcopy=False)
    if any(t.state == optuna.trial.TrialState.RUNNING for t in existing):
        parser.error("Study has running trials; stop/reconcile that run before resuming")
    finished = sum(t.state.is_finished() for t in existing)
    if args.trials < finished:
        parser.error(f"Study already has {finished} finished trials; budget must be >= that")
    feasible = [t for t in existing if t.state == optuna.trial.TrialState.COMPLETE
                and np.isfinite(t.value)
                and all(c <= 0 for c in t.user_attrs.get("constraints", [float("inf")]))]
    best = min(feasible, key=lambda t: t.value) if feasible else None
    if args.resume:
        record_path = output / f"resume_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        record_path.write_text(json.dumps({**settings, "previous_trials": finished,
                                          "sampler_rng_reseeded": True}, indent=2) + "\n")
    # A few conventional alternating patterns are starting hypotheses, not
    # claimed working gaits. Optuna is free to change every hinge's waveform.
    starting_patterns = [] if args.resume else [
        (0.7, [0, 1, 0, 1]), (1.0, [0, 0, 1, 1]),
        (1.5, [0, 1, 1, 0]), (2.0, [0, 1, 0, 1]),
    ]
    for frequency, pattern in starting_patterns:
        params = {"clock_hz": frequency if args.fixed_clock_hz is None else args.fixed_clock_hz}
        for hinge in range(output_size):
            params[f"amplitude_{hinge}"] = 0.8 if hinge % 2 == 0 else 0.6
            params[f"phase_{hinge}"] = float(
                (pattern[hinge // 2] * np.pi + (np.pi / 2 if hinge % 2 else 0)
                 + np.pi) % (2 * np.pi) - np.pi
            )
            params[f"bias_{hinge}"] = 0.0 if hinge % 2 == 0 else -0.4
        study.enqueue_trial(params)
    workers = min(args.workers, max(1, args.trials - finished))
    print(f"Output: {output}; completed={finished}; total budget={args.trials}; "
          f"workers={workers}", flush=True)
    if best is not None:
        print(f"Previous best feasible gait: trial {best.number}, distance={best.value:.4f}",
              flush=True)
    with (output / "gait_trials.csv").open("a" if args.resume else "w", newline="") as file:
        writer = csv.writer(file)
        if not args.resume:
            writer.writerow(["trial", "fitness", "distance", "upright_fraction", "feasible",
                             "core_contact_fraction", "airborne_fraction", "leg_support_fraction",
                             "low_core_shortfall", "mean_clearance"])
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=demo.initialise_worker,
            initargs=(1.0,),
        ) as pool:
            for start in range(finished, args.trials, workers):
                if best is not None and best.value <= args.target_fitness:
                    break
                trials = [study.ask() for _ in range(min(workers, args.trials - start))]
                parameters = [suggest_gait(trial, output_size, args.fixed_clock_hz)
                              for trial in trials]
                candidates = [(split_weights(gait_weights(p, input_size, output_size),
                                             input_size, output_size), p["clock_hz"])
                              for p in parameters]
                results = pool.map(evaluate_gait, candidates)
                for trial, (score, metrics) in zip(trials, results, strict=True):
                    gates = gait_constraints(score, metrics)
                    for name, value in zip(
                        ("upright_fraction", "final_up", "final_height", "progress"),
                        gates, strict=True,
                    ):
                        trial.set_constraint(name, float(value))
                    trial.set_user_attr("constraints", gates)
                    trial.set_user_attr("metrics", metrics)
                    trial.set_user_attr("search_seed", args.seed)
                    frozen = study.tell(trial, score)
                    feasible = all(value <= 0 for value in gates)
                    writer.writerow([
                        trial.number, score, metrics.get("distance", score),
                        metrics["upright_fraction"], feasible,
                        metrics.get("core_contact_fraction", 1.0),
                        metrics.get("airborne_fraction", 1.0), metrics.get("leg_support_fraction", 0.0),
                        metrics.get("low_core_shortfall", 1.0), metrics.get("mean_clearance", 0.0),
                    ])
                    file.flush()
                    if feasible and (best is None or score < best.value):
                        best = frozen
                        save_gait(output, best, input_size, output_size)
                        print(f"Best feasible gait: trial {trial.number}, score={score:.4f}, "
                              f"distance={metrics.get('distance', score):.4f}, "
                              f"core_contact={metrics.get('core_contact_fraction', 1.0):.0%}, "
                              f"airborne={metrics.get('airborne_fraction', 1.0):.0%}, "
                              f"clearance={metrics.get('mean_clearance', 0.0) * 100:.1f}cm", flush=True)
                print(f"Completed {start + len(trials)}/{args.trials} gait trials", flush=True)
                if best is not None and best.value <= args.target_fitness:
                    break
    if best is None:
        raise RuntimeError("No upright moving gait found; trials saved, ES not started")
    print(f"Gait checkpoint: {output / 'gait_best.npz'}; fitness={best.value:.4f}", flush=True)
    if args.train_after:
        subprocess.run([
            sys.executable, str(ROOT / "A2_template_2026.py"),
            "--warm-start", str(output / "gait_best.npz"),
            "--generations", str(args.generations), "--population", str(args.population),
            "--mutation", args.mutation_mode, "--workers", str(args.workers),
            "--seed", str(args.seed), "--output", str(output / "es"),
        ], check=True)


if __name__ == "__main__":
    main()
