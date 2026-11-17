"""fit_search: the fit search, many universes side by side on one GPU, one JSON line per run.

Contract: fit_search(results, ...) -> list of records. Every run is one fit_run in its own
process (`workers` of them at a time on the one GPU; each steps exactly as it would
alone, and a run that hangs or dies takes nothing with it). Phases, in order:
  defaults  the default settings and start, one run per seed in default_seeds
  random    random_points Latin-hypercube points over SEARCHED, on seed 1
  refine<r> for r in 1 .. refine_rounds: refine_count Gaussian steps round each of the
            refine_best best points searched so far, 0.15 of the unit cube wide in round
            1 and 0.7 times narrower each round after
  recheck   the recheck_top best searched points on every seed in recheck_seeds (the
            start's seed is never fitted: a setting is judged by its mean over seeds)
  sweep     the best searched point with one setting at a time moved to each of
            sweep_at (unit-cube places): where the score does not move, the fit cannot
            decide that setting
Every run is appended to `results` as it ends, with its phase, its id and its unit-cube
point; a run whose id is already in the file is skipped, so a stopped search resumes.
A run that raises, or does not finish within `seconds` of wall time (it is stopped
between steps, or killed after twice that), is recorded with score 0 and its error,
never dropped. `seconds` defaults to 3000: the 2026-11-17 search used 1800, about three
times a default universe's run, and cut eventful universes at 2,200 to 2,900 steps.
Settings searched, ranges fixed 2026-11-17 with the cell-grid engine and before any
search on it: pull_strength 1e-9 .. 1e-5, light_speed (the pull's speed) 0.25 .. 2,
push_strength 2e-8 .. 2e-4, squeeze_rate 0.01 .. 100, crossing_rate 1.3e-4 .. 1.3e-1
(all log); seal_thin 1.1 .. 5, collapse_thin 1.5 .. 10, tear_length 11 .. 40 (linear);
and the start's unevenness, linear in 0 .. 0.9 (owner, 2026-11-16). Every default lies
inside. Not searched, kept at their defaults: step_time, clock_packing, fluid_pressure.
No dark-energy or base-push setting is searched (bible rule 8).
"""
from __future__ import annotations

import json
import math
import multiprocessing
import pathlib
import time
import traceback

import numpy as np

from physics.fit.fit_run import fit_run

LOG_RANGES = {"pull_strength": (1e-9, 1e-5), "light_speed": (0.25, 2.0), "push_strength": (2e-8, 2e-4),
              "squeeze_rate": (0.01, 100.0), "crossing_rate": (1.3e-4, 1.3e-1)}
LINEAR_RANGES = {"seal_thin": (1.1, 5.0), "collapse_thin": (1.5, 10.0), "tear_length": (11.0, 40.0)}
START_RANGES = {"unevenness": (0.0, 0.9)}
START_NAMES = list(START_RANGES)
SEARCHED = list(LOG_RANGES) + list(LINEAR_RANGES) + START_NAMES


def _unit_to_overrides(unit, names=None):
    """Map a point of the unit cube to overrides; `names` are its axes (default SEARCHED, as many as the point has)."""
    values = {}
    for name, coordinate in zip(SEARCHED if names is None else names, unit):
        if name in LOG_RANGES:
            low, high = LOG_RANGES[name]
            values[name] = float(math.exp(math.log(low) + coordinate * (math.log(high) - math.log(low))))
        else:
            low, high = (LINEAR_RANGES | START_RANGES)[name]
            values[name] = float(low + coordinate * (high - low))
    return values


def _work(task, size, seconds, pipe):
    """In a worker process: run one point and send its record back."""
    try:
        record = fit_run(task["overrides"], task["seed"], *size, seconds=seconds)
    except Exception as error:  # a failed run is a result: recorded with its error, never hidden
        record = {"overrides": task["overrides"], "seed": task["seed"], "score": 0.0, "E": 0.0, "S": 0.0,
                  "error": repr(error), "trace": traceback.format_exc()[-800:]}
    pipe.send(record)
    pipe.close()


def _killed(task, seconds):
    """The record of a run whose process gave no answer."""
    return {"overrides": task["overrides"], "seed": task["seed"], "score": 0.0, "E": 0.0, "S": 0.0,
            "error": f"no answer within {2 * seconds} s; process stopped"}


def _write(results, task, record):
    """Append one run's record with its task's phase, id and unit point; print one progress line."""
    record.update(phase=task["phase"], id=task["id"], unit=task["unit"])
    with results.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(task["id"], round(record["score"], 4), round(record.get("seconds", 0.0)), record.get("error", ""), flush=True)


def _run_all(results, tasks, size, workers, seconds):
    """Run the tasks not yet in the results file, `workers` processes at a time."""
    done = {record["id"] for record in _load(results)}
    waiting = [task for task in tasks if task["id"] not in done][::-1]
    context = multiprocessing.get_context("spawn")
    running = []
    while waiting or running:
        while waiting and len(running) < workers:
            task = waiting.pop()
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(target=_work, args=(task, size, seconds, sender), daemon=True)
            process.start()
            sender.close()
            running.append((task, process, receiver, time.time()))
        still = []
        for task, process, receiver, started in running:
            if receiver.poll(0.05):
                try:
                    _write(results, task, receiver.recv())
                except EOFError:
                    _write(results, task, {**_killed(task, seconds), "error": "the process died without an answer"})
                process.join()
            elif not process.is_alive() and not receiver.poll(0.05):
                _write(results, task, {**_killed(task, seconds), "error": "the process died without an answer"})
            elif seconds is not None and time.time() - started > 2 * seconds:
                process.kill()
                process.join()
                _write(results, task, _killed(task, seconds))
            else:
                still.append((task, process, receiver, started))
        running = still


def _load(results):
    """All records written so far."""
    if not results.exists():
        return []
    return [json.loads(line) for line in results.read_text().splitlines() if line.strip()]


def _best(records, count):
    """The `count` highest-scoring searched records (random and refine phases)."""
    searched = [record for record in records if record["phase"] == "random" or record["phase"].startswith("refine")]
    return sorted(searched, key=lambda record: -record["score"])[:count]


def _task(phase, index, unit, seed):
    """One run to do: its id, phase, unit point (None: the defaults), overrides and seed."""
    return {"id": f"{phase}:{index}:{seed}", "phase": phase, "seed": seed,
            "unit": None if unit is None else [float(value) for value in unit],
            "overrides": {} if unit is None else _unit_to_overrides(unit)}


def fit_search(results, cells=24, steps=3000, every=100, workers=8, seconds=3000.0, default_seeds=(1, 2, 3),
               random_points=100, refine_rounds=3, refine_best=5, refine_count=5, recheck_top=4, recheck_seeds=(2, 3),
               sweep_at=(0.1, 0.37, 0.63, 0.9), lattice_seed=20261117):
    """Run the search's runs not yet in `results` and return every record in the file."""
    results = pathlib.Path(results)
    results.parent.mkdir(parents=True, exist_ok=True)
    size = (cells, steps, every)
    generator = np.random.default_rng(lattice_seed)
    lattice = np.stack([(generator.permutation(random_points) + generator.random(random_points)) / max(random_points, 1)
                        for _ in SEARCHED], axis=1)
    tasks = [_task("defaults", 0, None, seed) for seed in default_seeds]
    tasks += [_task("random", index, unit, 1) for index, unit in enumerate(lattice)]
    _run_all(results, tasks, size, workers, seconds)
    for round_index in range(1, refine_rounds + 1):
        steps_from = np.random.default_rng((lattice_seed, round_index))
        width = 0.15 * 0.7 ** (round_index - 1)
        tasks = []
        for record in _best(_load(results), refine_best):
            for _ in range(refine_count):
                unit = np.clip(np.asarray(record["unit"]) + steps_from.normal(0.0, width, len(SEARCHED)), 0.0, 1.0)
                tasks.append(_task(f"refine{round_index}", len(tasks), unit, 1))
        _run_all(results, tasks, size, workers, seconds)
    tasks = [_task("recheck", rank, record["unit"], seed)
             for rank, record in enumerate(_best(_load(results), recheck_top)) for seed in recheck_seeds]
    _run_all(results, tasks, size, workers, seconds)
    tasks = []
    for record in _best(_load(results), 1 if sweep_at else 0):
        for axis in range(len(SEARCHED)):
            for place in sweep_at:
                unit = list(record["unit"])
                unit[axis] = place
                tasks.append(_task("sweep", f"{SEARCHED[axis]}@{place}", unit, 1))
    _run_all(results, tasks, size, workers, seconds)
    return _load(results)
