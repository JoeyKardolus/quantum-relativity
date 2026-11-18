"""Run search: the fit search's phases in order, one JSON line per run, resumable.

Contract: `run_search(results, plan=SearchPlan(), minutes=None)` runs the runs not yet in
the file `results` and returns every record in it. Phases, in order:
  defaults  the default settings, one run per seed in plan.default_seeds
  random    the plan's Latin-hypercube points over SEARCHED, on seed 1 (only the first
            plan.random_run of them when that is given: a plain random sample)
  refine<r> for r in 1 .. plan.refine_rounds: plan.refine_count Gaussian steps
            (`refine_width` wide) round each of the plan.refine_best best points
            searched so far
  hold      a short run's score can be a runaway caught midway, so the plan.hold_top
            best searched points whose hold factor is above CALM are run again for
            plan.hold_steps on seed 1; those that hold (`find_holders`) are run on
            plan.hold_seeds too. The fitted point is the holder with the best mean long
            score; with no holder nothing is fitted
  sweep     the fitted point (long runs), or without one the best short point (short
            runs), with one setting at a time moved to each of plan.sweep_at, on every
            seed in plan.sweep_seeds: where the score does not move, the fit cannot
            decide that setting
Every run is appended to `results` as it ends; a run whose id is already there is
skipped, so a stopped search resumes. With `minutes`, no new run is started after that
many minutes: the runs under way are finished and recorded and the search returns where
it stands, to be run again. A run that raises, hangs or dies is recorded with score 0
and its error, never dropped.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from physics.fit.search.definitions.latin_hypercube import latin_hypercube
from physics.fit.search.definitions.make_task import make_task
from physics.fit.search.definitions.point_holds import CALM
from physics.fit.search.definitions.ranges import SEARCHED
from physics.fit.search.definitions.refine_width import refine_width
from physics.fit.search.definitions.search_plan import SearchPlan
from physics.fit.search.definitions.searched_phase import searched_phase
from physics.fit.search.logic.find_holders import find_holders
from physics.fit.search.logic.load_records import load_records
from physics.fit.search.logic.run_tasks import run_tasks


def _best(records: list, count: int | None) -> list:
    """The `count` highest-scoring searched records, all of them for None."""
    searched = [record for record in records if searched_phase(record["phase"])]
    return sorted(searched, key=lambda record: -record["score"])[:count]


def _first_tasks(plan: SearchPlan) -> list:
    """The defaults on every default seed, then the random points on seed 1."""
    points = latin_hypercube(plan.random_points, len(SEARCHED), plan.lattice_seed)[:plan.random_run]
    return ([make_task("defaults", 0, None, seed) for seed in plan.default_seeds]
            + [make_task("random", index, unit, 1) for index, unit in enumerate(points)])


def _refine_tasks(plan: SearchPlan, round_index: int, records: list) -> list:
    """Gaussian steps round each of the best points searched so far, clipped to the cube."""
    steps_from = np.random.default_rng((plan.lattice_seed, round_index))
    tasks: list = []
    for record in _best(records, plan.refine_best):
        for _ in range(plan.refine_count):
            moved = np.asarray(record["unit"]) + steps_from.normal(0.0, refine_width(round_index), len(SEARCHED))
            tasks.append(make_task(f"refine{round_index}", len(tasks), np.clip(moved, 0.0, 1.0), 1))
    return tasks


def _calm_best(plan: SearchPlan, records: list) -> list:
    """The best searched records whose short run stayed calm, at most plan.hold_top of them."""
    return [record for record in _best(records, None) if record.get("hold", 0.0) > CALM][:plan.hold_top]


def _hold_tasks(plan: SearchPlan, records: list) -> list:
    """The calm best points again on seed 1, and every holder among them on the further seeds."""
    calm = _calm_best(plan, records)
    ranks = {json.dumps(record["unit"]): rank for rank, record in enumerate(calm)}
    holders = [point for point in find_holders(records) if point["holds"]]
    return ([make_task("hold", rank, record["unit"], 1) for rank, record in enumerate(calm)]
            + [make_task("hold", ranks[json.dumps(point["unit"])], point["unit"], seed)
               for point in holders for seed in plan.hold_seeds])


def _sweep_tasks(plan: SearchPlan, centre) -> list:
    """The centre point with one setting at a time moved to each sweep place, on every sweep seed."""
    tasks = []
    for axis, name in enumerate(SEARCHED if centre else []):
        for place in plan.sweep_at:
            unit = list(centre)
            unit[axis] = place
            tasks += [make_task("sweep", f"{name}@{place}", unit, seed) for seed in plan.sweep_seeds]
    return tasks


def _stages(plan: SearchPlan):
    """The search's stages in order: per stage a function of the records so far giving (tasks, size, seconds)."""
    short = ((plan.cells, plan.steps, plan.every), plan.seconds)
    long = ((plan.cells, plan.hold_steps, plan.long_every()), plan.long_seconds())
    yield lambda records: (_first_tasks(plan), *short)
    for round_index in range(1, plan.refine_rounds + 1):
        yield lambda records, round_index=round_index: (_refine_tasks(plan, round_index, records), *short)
    for _ in ("the calm best on seed 1", "the holders found among them on the further seeds"):
        yield lambda records: (_hold_tasks(plan, records), *long)
    yield lambda records: _sweep_stage(plan, records, short, long)


def _sweep_stage(plan: SearchPlan, records: list, short: tuple, long: tuple) -> tuple:
    """The sweep round the best holder at the long size, or round the best short point at the short size."""
    held = [point for point in find_holders(records) if point["holds"]]
    if held:
        return (_sweep_tasks(plan, held[0]["unit"]), *long)
    best = _best(records, 1)
    return (_sweep_tasks(plan, best[0]["unit"] if best else None), *short)


def run_search(results, plan: SearchPlan | None = None, minutes: float | None = None) -> list:
    """Run the search's runs not yet in `results` and return every record in the file."""
    plan = SearchPlan() if plan is None else plan
    until = None if minutes is None else time.time() + 60.0 * minutes
    results = pathlib.Path(results)
    results.parent.mkdir(parents=True, exist_ok=True)
    for stage in _stages(plan):
        tasks, size, seconds = stage(load_records(results))
        if not run_tasks(results, tasks, size, plan.workers, seconds, until, plan.device):
            break
    return load_records(results)
