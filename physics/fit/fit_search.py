"""fit_search: the fit search, many universes side by side on one GPU, one JSON line per run.

Contract: fit_search(results, ...) -> list of records. Every run is one fit_run in its own
process (`workers` of them at a time on the one GPU; each steps exactly as it would
alone, and a run that hangs or dies takes nothing with it). Phases, in order:
  defaults  the default settings and start, one run per seed in default_seeds
  random    random_points Latin-hypercube points over SEARCHED, on seed 1 (only the first
            random_run of them when that is given: a plain random sample of the cube)
  refine<r> for r in 1 .. refine_rounds: refine_count Gaussian steps round each of the
            refine_best best points searched so far, 0.15 of the unit cube wide in round
            1 and 0.7 times narrower each round after
  hold      a short run's score can be a runaway caught midway (measured: a point that
            scored 0.23 at 5,000 steps tore itself apart at 5,600), so the hold_top
            best searched points whose hold factor is above 0.9 are run again for
            hold_steps on seed 1; those that hold (fit_holders: at least half the short
            score kept, still calm) are run for hold_steps on hold_seeds too. The fitted
            point is the holder with the best mean long score; with no holder nothing
            is fitted
  sweep     the fitted point (hold_steps a run), or without one the best short point
            (steps a run), with one setting at a time moved to each of sweep_at
            (unit-cube places), on every seed in sweep_seeds: where the score does not
            move, the fit cannot decide that setting
Every run is appended to `results` as it ends, with its phase, its id and its unit-cube
point; a run whose id is already in the file is skipped, so a stopped search resumes.
With `minutes`, no new run is started after that many minutes: the runs under way are
finished and recorded and the search returns where it stands, to be run again.
A run that raises, or does not finish within `seconds` of wall time (it is stopped
between steps, or killed after twice that), is recorded with score 0 and its error,
never dropped.
Searched, every rule's strength in physics.engine Settings and the start, ranges fixed
on 2026-11-17 before any search on the cell grid (each default lies inside):
  log     pull_strength 1e-9 .. 1e-4, light_speed (the pull's speed) 0.1 .. 4,
          push_strength 2e-9 .. 2e-2, squeeze_rate 0.01 .. 100, crossing_rate 1.3e-4 ..
          1.3e-1, seal_thin 1.05 .. 10, collapse_thin 1.2 .. 30, tear_length 11 .. 1e5
          (a volume at the engine's floor is 30,000 long, so the top of the range is a
          tear that never comes), clock_packing 1 .. 1000, fluid_pressure 0.05 .. 5;
          the start: joint_length 0.1 .. 10 (the joints' size constant against the
          volumes'; the volumes' own, 10 long, is the unit of length and stays),
          box_quanta 1000 .. 10000 and joint_quanta 30 .. 300 (the start amounts per box;
          joints stay thin mortar holding the smaller share)
  linear  the start's unevenness of the volumes 0 .. 0.9 and of the joints 0 .. 0.9
Not searched: step_time (the counter's step, no rule), the unevenness's slope (-3:
ripples at every size alike) and the jitter inside a cell (0.1). No dark-energy or
base-push setting exists (bible rule 8).
"""
from __future__ import annotations

import json
import math
import multiprocessing
import pathlib
import time
import traceback

import numpy as np

from physics.fit.fit_holders import CALM, fit_holders
from physics.fit.fit_run import fit_run

LOG_RANGES = {"pull_strength": (1e-9, 1e-4), "light_speed": (0.1, 4.0), "push_strength": (2e-9, 2e-2),
              "squeeze_rate": (0.01, 100.0), "crossing_rate": (1.3e-4, 1.3e-1), "seal_thin": (1.05, 10.0),
              "collapse_thin": (1.2, 30.0), "tear_length": (11.0, 1e5), "clock_packing": (1.0, 1000.0),
              "fluid_pressure": (0.05, 5.0), "joint_length": (0.1, 10.0), "box_quanta": (1000.0, 10000.0),
              "joint_quanta": (30.0, 300.0)}
LINEAR_RANGES = {"unevenness": (0.0, 0.9), "joint_unevenness": (0.0, 0.9)}
SEARCHED = list(LOG_RANGES) + list(LINEAR_RANGES)


def _unit_to_overrides(unit, names=None):
    """Map a point of the unit cube to overrides; `names` are its axes (default SEARCHED, as many as the point has)."""
    values = {}
    for name, coordinate in zip(SEARCHED if names is None else names, unit):
        if name in LOG_RANGES:
            low, high = LOG_RANGES[name]
            values[name] = float(math.exp(math.log(low) + coordinate * (math.log(high) - math.log(low))))
        else:
            low, high = LINEAR_RANGES[name]
            values[name] = float(low + coordinate * (high - low))
    return values


def _work(task, size, seconds, pipe):
    """In a worker process: run one point and send its record back."""
    try:
        record = fit_run(task["overrides"], task["seed"], *size, seconds=seconds)
        if not math.isfinite(record["score"]):
            record.update(score=0.0, E=0.0, S=0.0, error="the score is not a number: the run left the numbers' range")
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


def _run_all(results, tasks, size, workers, seconds, until=None):
    """Run the tasks not yet in the results file, `workers` processes at a time; False when `until` cut it short."""
    done = {record["id"] for record in _load(results)}
    waiting = [task for task in tasks if task["id"] not in done][::-1]
    context = multiprocessing.get_context("spawn")
    running, cut = [], False
    while waiting or running:
        if until is not None and time.time() > until and waiting:
            waiting = []
            cut = True
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
    return not cut


def _load(results):
    """All records written so far."""
    if not results.exists():
        return []
    return [json.loads(line) for line in results.read_text().splitlines() if line.strip()]


def _best(records, count):
    """The `count` highest-scoring searched records (random and refine phases), all of them for None."""
    searched = [record for record in records if record["phase"] == "random" or record["phase"].startswith("refine")]
    return sorted(searched, key=lambda record: -record["score"])[:count]


def _task(phase, index, unit, seed):
    """One run to do: its id, phase, unit point (None: the defaults), overrides and seed."""
    return {"id": f"{phase}:{index}:{seed}", "phase": phase, "seed": seed,
            "unit": None if unit is None else [float(value) for value in unit],
            "overrides": {} if unit is None else _unit_to_overrides(unit)}


def fit_search(results, cells=24, steps=5000, every=100, workers=12, seconds=240.0, default_seeds=(1, 2, 3, 4, 5, 6),
               random_points=3000, random_run=None, refine_rounds=3, refine_best=12, refine_count=16, hold_top=100, hold_steps=20000,
               hold_seeds=(2, 3), sweep_at=(0.1, 0.3, 0.5, 0.7, 0.9), sweep_seeds=(1, 2), lattice_seed=20261117,
               minutes=None):
    """Run the search's runs not yet in `results` and return every record in the file."""
    until = None if minutes is None else time.time() + 60.0 * minutes
    results = pathlib.Path(results)
    results.parent.mkdir(parents=True, exist_ok=True)
    size = (cells, steps, every)
    generator = np.random.default_rng(lattice_seed)
    lattice = np.stack([(generator.permutation(random_points) + generator.random(random_points)) / max(random_points, 1)
                        for _ in SEARCHED], axis=1)
    tasks = [_task("defaults", 0, None, seed) for seed in default_seeds]
    tasks += [_task("random", index, unit, 1) for index, unit in enumerate(lattice[:random_run])]
    if not _run_all(results, tasks, size, workers, seconds, until):
        return _load(results)
    for round_index in range(1, refine_rounds + 1):
        steps_from = np.random.default_rng((lattice_seed, round_index))
        width = 0.15 * 0.7 ** (round_index - 1)
        tasks = []
        for record in _best(_load(results), refine_best):
            for _ in range(refine_count):
                unit = np.clip(np.asarray(record["unit"]) + steps_from.normal(0.0, width, len(SEARCHED)), 0.0, 1.0)
                tasks.append(_task(f"refine{round_index}", len(tasks), unit, 1))
        if not _run_all(results, tasks, size, workers, seconds, until):
            return _load(results)
    long = (cells, hold_steps, max(every * hold_steps // max(steps, 1), 1))
    patience = None if seconds is None else seconds * hold_steps / max(steps, 1)
    calm = [record for record in _best(_load(results), None) if record.get("hold", 0.0) > CALM][:hold_top]
    ranks = {json.dumps(record["unit"]): rank for rank, record in enumerate(calm)}
    tasks = [_task("hold", rank, record["unit"], 1) for rank, record in enumerate(calm)]
    if not _run_all(results, tasks, long, workers, patience, until):
        return _load(results)
    holders = [point for point in fit_holders(_load(results)) if point["holds"]]
    tasks = [_task("hold", ranks[json.dumps(point["unit"])], point["unit"], seed) for point in holders for seed in hold_seeds]
    if not _run_all(results, tasks, long, workers, patience, until):
        return _load(results)
    held = [point for point in fit_holders(_load(results)) if point["holds"]]
    centre = held[0]["unit"] if held else next((record["unit"] for record in _best(_load(results), 1)), None)
    tasks = []
    for axis in range(len(SEARCHED) if centre and sweep_at else 0):
        for place in sweep_at:
            unit = list(centre)
            unit[axis] = place
            tasks += [_task("sweep", f"{SEARCHED[axis]}@{place}", unit, seed) for seed in sweep_seeds]
    _run_all(results, tasks, long if held else size, workers, patience if held else seconds, until)
    return _load(results)
