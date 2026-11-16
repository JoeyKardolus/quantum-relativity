"""fit_search: the fit search, run by run on one GPU, one JSON line per run.

Contract: fit_search(results, ...) -> list of records. Phases, in order: defaults (one run per
seed in default_seeds), random (random_points Latin-hypercube points on seed 1), refine
(refine_count Gaussian steps of 0.15 unit-cube width round each of the refine_best best),
recheck (the recheck_top best searched points on every seed in recheck_seeds). Every run is
appended to `results` as it ends; a phase already in the file is skipped, so a stopped search
resumes. A run that raises is recorded with score -2 and its error, never dropped.
Settings fitted, ranges fixed for the whole-quanta engine (2026-11-16) before any search
on it: pull_strength, push, collapse_packing, crossing_rate (log, each a decade or more round
its default), push_steepness and seal_squeeze (linear; seal 0.2 .. 2, round its default
1; owner, 2026-11-16: the collapse is set by packing, so the seal is no longer a share of
it). Searched beside them, the start's two
unevennesses (owner, 2026-11-16, decision D), separate settings of
physics.engine.new_universe: unevenness (energy volumes) and joint_unevenness (joints), each
linear in 0 .. 0.9 (owner, 2026-11-16: widened from 0 .. 0.6 so the fit decides, the
owner holding no opinion; the defaults 0.3 and 0 lie inside). The unit cube has SEARCHED axes: NAMES (the engine settings, which the stand-in
also reads) then START_NAMES. The cell engine's search (attraction, push_power,
compressions) cannot be compared with these.
No dark-energy or base-push setting is searched (bible rule 8: expansion is drift from
imperfect transfer, nothing pushes space apart).
"""
from __future__ import annotations

import json
import math
import pathlib
import traceback

import numpy as np

from physics.fit.fit_run import fit_run

LOG_RANGES = {"pull_strength": (1e-8, 1e-6), "push": (0.05, 5.0), "collapse_packing": (2.7e3, 2.7e5),
              "crossing_rate": (4e-4, 4e-2)}
LINEAR_RANGES = {"push_steepness": (2.0, 10.0), "seal_squeeze": (0.2, 2.0)}
NAMES = list(LOG_RANGES) + list(LINEAR_RANGES)
START_RANGES = {"unevenness": (0.0, 0.9), "joint_unevenness": (0.0, 0.9)}
START_NAMES = list(START_RANGES)
SEARCHED = NAMES + START_NAMES


def _unit_to_overrides(unit):
    """Map a point of the unit cube (NAMES, or all SEARCHED axes) to overrides."""
    values = {}
    for name, coordinate in zip(SEARCHED, unit):
        if name in LOG_RANGES:
            low, high = LOG_RANGES[name]
            values[name] = float(math.exp(math.log(low) + coordinate * (math.log(high) - math.log(low))))
        else:
            low, high = (LINEAR_RANGES | START_RANGES)[name]
            values[name] = float(low + coordinate * (high - low))
    return values


def _run(results, size, phase, unit, overrides, seed):
    """Run one point, append its record to the results file, print one progress line."""
    try:
        record = fit_run(overrides, seed, *size)
    except Exception as error:  # a failed run is a result: recorded with its error, never hidden
        record = {"overrides": overrides, "seed": seed, "score": -2.0, "error": repr(error),
                  "trace": traceback.format_exc()[-800:]}
    record["phase"] = phase
    record["unit"] = None if unit is None else [float(value) for value in unit]
    with results.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(phase, seed, round(record["score"], 4), record.get("error", ""), flush=True)
    return record


def _load(results):
    """All records written so far."""
    if not results.exists():
        return []
    return [json.loads(line) for line in results.read_text().splitlines() if line.strip()]


def _best(records, phases, count):
    """The `count` highest-scoring records of the given phases."""
    return sorted([record for record in records if record["phase"] in phases], key=lambda record: -record["score"])[:count]


def fit_search(results, cells=7, boxes=10, steps=120, every=6, default_seeds=(1, 1, 1, 2, 3), random_points=80,
               refine_best=5, refine_count=8, recheck_top=5, recheck_seeds=(1, 2, 3), lattice_seed=20261116):
    """Run the search's phases not yet in `results` and return every record in the file."""
    results = pathlib.Path(results)
    results.parent.mkdir(parents=True, exist_ok=True)
    size = (cells, boxes, steps, every)
    phases = {record["phase"] for record in _load(results)}
    if "defaults" not in phases:
        for seed in default_seeds:
            _run(results, size, "defaults", None, {}, seed)
    generator = np.random.default_rng(lattice_seed)
    lattice = np.stack([(generator.permutation(random_points) + generator.random(random_points)) / max(random_points, 1)
                        for _ in SEARCHED], axis=1)
    if "random" not in phases:
        for unit in lattice:
            _run(results, size, "random", unit, _unit_to_overrides(unit), 1)
    if "refine" not in phases:
        for record in _best(_load(results), ("random",), refine_best):
            for _ in range(refine_count):
                unit = np.clip(np.asarray(record["unit"]) + generator.normal(0, 0.15, len(SEARCHED)), 0, 1)
                _run(results, size, "refine", unit, _unit_to_overrides(unit), 1)
    if "recheck" not in phases:
        for record in _best(_load(results), ("random", "refine"), recheck_top):
            for seed in recheck_seeds:
                _run(results, size, "recheck", record["unit"], record["overrides"], seed)
    return _load(results)
