"""fit_run: one fit-set universe, stepped, measured inside and scored on expansion shape and structure.

Contract: fit_run(overrides, seed, cells, boxes, steps, every, seconds=None) -> dict.
Builds physics.engine new_universe(cells, boxes, seed) with margin 0, which must wrap (an
even `cells` and an even side): the universe has no edge, so space is measured inside it
(fit_measure: boxes between lumps), never where it meets nothing. The two start
unevennesses (unevenness, joint_unevenness) are taken from `overrides` when there
(new_universe's defaults otherwise), every other override is a Settings field. Samples
fit_measure at the start and every `every` steps, with blocks two start cells across.
Time is the observer's (bible: time is local): tau adds up, sample to sample, the
counter time times the mean of the two samples' observer_rate, the clock where the
typical quantum sits. An astronomer is made of energy, so the fit times the expansion on
that clock; the counter-time shape is returned beside it (`counter`, the same score on
the step counter) so the clock's part can be seen. Whether the densest observers see
more speeding up than the emptiest is the timescape scene's unfitted test.
Returns the overrides, sizes, every fit_score part, the series (one list per measured
name, plus "step", "time", "tau"), wall `seconds`, and `unfinished`: True when
`seconds` of wall time ran out before `steps` (the score is then 0 and "error" says so:
a universe that cannot be stepped to the end was not measured). No dark-energy or
base-push setting exists to pass (bible rule 8). Raises ValueError for a universe that
does not wrap; the engine's errors propagate.
"""
from __future__ import annotations

import time

from physics.engine import Settings, new_universe, step
from physics.fit.fit_measure import fit_measure
from physics.fit.fit_score import fit_score

_START_KEYS = ("unevenness", "joint_unevenness")


def fit_run(overrides: dict, seed: int, cells: int, boxes: int, steps: int, every: int,
            seconds: float | None = None) -> dict:
    """Run one wrapping universe from the overrides' start and settings; return its score, parts and series."""
    started = time.time()
    settings = Settings(**{name: value for name, value in overrides.items() if name not in _START_KEYS})
    start = {name: value for name, value in overrides.items() if name in _START_KEYS}
    universe = new_universe(cells, boxes, seed, **start)
    if not universe.wrap:
        raise ValueError(f"the fit needs a wrapping universe (even cells and side), got {cells} cells of {boxes}")
    block = 2 * (boxes + 1)
    samples = [{"step": 0, "time": 0.0, "tau": 0.0, **fit_measure(universe, settings, block)}]
    unfinished = False
    for taken in range(1, steps + 1):
        step(universe, settings)
        if taken % every == 0 or taken == steps:
            measured = fit_measure(universe, settings, block)
            before = samples[-1]
            passed = (taken - before["step"]) * settings.step_time
            samples.append({"step": taken, "time": taken * settings.step_time,
                            "tau": before["tau"] + passed * 0.5 * (before["observer_rate"] + measured["observer_rate"]),
                            **measured})
        if seconds is not None and time.time() - started > seconds and taken < steps:
            unfinished = True
            break
    record = {"overrides": overrides, "seed": seed, "cells": cells, "boxes": boxes, "steps": steps,
              "steps_done": samples[-1]["step"], "unfinished": unfinished, "seconds": time.time() - started,
              "series": {name: [sample[name] for sample in samples] for name in samples[0]}}
    if unfinished or len(samples) < 4:
        return {**record, "score": 0.0, "E": 0.0, "S": 0.0,
                "error": f"unfinished: {samples[-1]['step']} of {steps} steps in {seconds} s"}
    counter = fit_score([{**sample, "tau": sample["time"]} for sample in samples])
    return {**record, **fit_score(samples), "counter": {name: counter[name] for name in ("score", "E", "dip", "r1", "r2", "r3")}}
