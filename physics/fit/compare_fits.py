"""Compare fits: the gradient fit against random search on the side-by-side stand-in, both checked on the real fit score.

Contract: `compare_fits(out_dir, cells=8, boxes=10, random_points=64, starts=4, iterations=15,
rate=0.05, steps=24, every=3, top=2, seeds=(1, 2, 3), lattice_seed=20261116)` runs, in order,
and writes each part to out_dir as JSON:
  random.json    random_points Latin-hypercube points (fit_search's construction and seed)
                 scored side by side on the stand-in (stand_in_search), with its wall time
  gradient.json  gradient_fit from `starts` points: the defaults' unit point and the first
                 starts - 1 of the same lattice, `iterations` Adam steps, with wall times
  real.jsonl     the real fit score (fit_run: the engine itself, whole quanta, moving cells and
                 collapse, the observer's clock, 120 steps every 6, wrapping) on every seed of `seeds`, for
                 the defaults and the `top` best points of each method by stand-in score
                 (gradient: each start's last point)
and returns a summary dict: for each method its evaluations, seconds, best stand-in score and
the real scores (mean and each seed) of its top points; and the defaults' real scores. The
start's seed is 1 for the stand-in (new_universe(cells, boxes, 1, margin=0)); it is never
fitted. Every setting, range, size and seed here is fixed before a run (fit/CLAUDE.md).
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from physics.engine import Settings, host_boxes, new_universe
from physics.fit.fit_run import fit_run
from physics.fit.fit_search import LINEAR_RANGES, LOG_RANGES, NAMES, _unit_to_overrides
from physics.fit.gradient_fit import gradient_fit
from physics.fit.stand_in_search import stand_in_search

_REAL_STEPS = 120
_REAL_EVERY = 6


def compare_fits(out_dir, cells: int = 8, boxes: int = 10, random_points: int = 64, starts: int = 4,
                 iterations: int = 15, rate: float = 0.05, steps: int = 24, every: int = 3, top: int = 2,
                 seeds=(1, 2, 3), lattice_seed: int = 20261116) -> dict:
    """Random search and the gradient fit on the stand-in, then their best points on the real fit score."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = host_boxes(new_universe(cells, boxes, 1, margin=0))["quanta"]
    generator = np.random.default_rng(lattice_seed)
    lattice = np.stack([(generator.permutation(random_points) + generator.random(random_points)) / random_points
                        for _ in NAMES], axis=1)
    searched = stand_in_search(start, lattice, steps, every, block=boxes)
    random_record = {"points": lattice.tolist(), "score": searched["score"].tolist(),
                     "parts": searched["parts"].tolist(), "seconds": searched["seconds"]}
    (out_dir / "random.json").write_text(json.dumps(random_record))
    climb_from = np.vstack([_default_unit()[None], lattice[:starts - 1]])
    climbed = gradient_fit(start, climb_from, iterations, rate, steps, every, block=boxes)
    (out_dir / "gradient.json").write_text(json.dumps(climbed))
    final = climbed[-1]
    random_best = np.argsort(-searched["score"])[:top]
    gradient_best = np.argsort(-np.asarray(final["score"]))[:top]
    candidates = [("defaults", None, None)]
    candidates += [("random", lattice[index], float(searched["score"][index])) for index in random_best]
    candidates += [("gradient", np.asarray(final["units"][index]), float(final["score"][index]))
                   for index in gradient_best]
    real = []
    with (out_dir / "real.jsonl").open("w") as handle:
        for method, unit, stand_in in candidates:
            overrides = {} if unit is None else _unit_to_overrides(unit, NAMES)
            scores = [fit_run(overrides, seed, cells, boxes, _REAL_STEPS, _REAL_EVERY)["score"] for seed in seeds]
            entry = {"method": method, "unit": None if unit is None else [float(value) for value in unit],
                     "overrides": overrides, "stand_in": stand_in, "real": scores, "real_mean": float(np.mean(scores))}
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
            real.append(entry)
    seconds_gradient = sum(record["seconds"] for record in climbed)
    return {"random": {"evaluations": random_points, "seconds": searched["seconds"],
                       "best_stand_in": float(searched["score"].max()),
                       "real": [entry for entry in real if entry["method"] == "random"]},
            "gradient": {"evaluations": starts * iterations + starts, "seconds": seconds_gradient,
                         "start_stand_in": climbed[0]["score"], "best_stand_in": float(max(final["score"])),
                         "real": [entry for entry in real if entry["method"] == "gradient"]},
            "defaults": [entry for entry in real if entry["method"] == "defaults"][0]}


def _default_unit() -> np.ndarray:
    """The defaults' place in the unit cube."""
    defaults = Settings()
    values = {name: getattr(defaults, name) for name in NAMES}
    unit = []
    for name in NAMES:
        if name in LOG_RANGES:
            low, high = LOG_RANGES[name]
            unit.append((math.log(values[name]) - math.log(low)) / (math.log(high) - math.log(low)))
        else:
            low, high = LINEAR_RANGES[name]
            unit.append((values[name] - low) / (high - low))
    return np.asarray(unit)
