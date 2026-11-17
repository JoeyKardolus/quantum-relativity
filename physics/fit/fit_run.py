"""fit_run: one fit-set universe, stepped, measured inside and scored on expansion shape and structure.

Contract: fit_run(overrides, seed, cells, steps, every, seconds=None) -> dict.
Builds physics.engine new_universe(cells, seed): it wraps, so the universe has no edge
and everything is measured inside it. The start's names in `overrides` (START_KEYS:
unevenness, joint_unevenness, box_quanta, joint_quanta, joint_length) go to new_universe,
every other override is a Settings field. Samples fit_measure and fit_snapshot at the
start and every `every` steps.
Time is the observer's (bible: time is local): tau adds up, sample to sample, the
counter time times the mean of the two samples' observer_rate, the clock where the
typical quantum sits. An astronomer is made of energy, so the fit times the expansion on
that clock; the counter-time score is returned beside it (`counter`) so the clock's
part can be seen.
A run is ended early, between steps, when the universe is over: more than two tears per
start volume, fewer than 5% of the start's volumes alive, or more than 95% of the size
spent (`ended` names which). Its score is the same formula on the samples so far, whose
hold factor is then under 0.05: an economy, since such universes take seconds a step.
Returns the overrides, sizes, every fit_score part, the series (one list per measured
name, plus "step", "time", "tau", "gap"), `held_share`, wall `seconds`, and
`unfinished`: True when `seconds` of wall time ran out before `steps` (the score is then
0 and "error" says so: a universe that cannot be stepped to the end was not measured).
No dark-energy or base-push setting exists to pass (bible rule 8). The engine's errors
propagate.
"""
from __future__ import annotations

import time

from physics.engine import Settings, new_universe, step
from physics.fit.fit_gap import fit_gap
from physics.fit.fit_measure import fit_measure
from physics.fit.fit_score import fit_score
from physics.fit.fit_snapshot import fit_snapshot

START_KEYS = ("unevenness", "joint_unevenness", "box_quanta", "joint_quanta", "joint_length")
_WHOLE = ("box_quanta", "joint_quanta")


def fit_run(overrides: dict, seed: int, cells: int, steps: int, every: int, seconds: float | None = None) -> dict:
    """Run one wrapping universe from the overrides' start and settings; return its score, parts and series."""
    started = time.time()
    settings = Settings(**{name: value for name, value in overrides.items() if name not in START_KEYS})
    start = {name: int(round(value)) if name in _WHOLE else value for name, value in overrides.items()
             if name in START_KEYS}
    universe = new_universe(cells, seed, **start)
    volumes = (universe.side // 2) ** 3
    samples = [{"step": 0, "time": 0.0, "tau": 0.0, **fit_measure(universe, settings)}]
    snapshots = [fit_snapshot(universe)]
    unfinished, ended = False, ""
    for taken in range(1, steps + 1):
        step(universe, settings)
        torn_out = universe.tears > 2 * volumes
        if taken % every == 0 or taken == steps or torn_out:
            measured = fit_measure(universe, settings)
            before = samples[-1]
            passed = (taken - before["step"]) * settings.step_time
            samples.append({"step": taken, "time": taken * settings.step_time,
                            "tau": before["tau"] + passed * 0.5 * (before["observer_rate"] + measured["observer_rate"]),
                            **measured})
            snapshots.append(fit_snapshot(universe, snapshots[-1]))
            ended = ("more than two tears per start volume" if torn_out else
                     "fewer than 5% of the start's volumes alive" if measured["alive"] < 0.05 else
                     "more than 95% of the size spent" if measured["spent_share"] > 0.95 else "")
            if ended:
                break
        if seconds is not None and time.time() - started > seconds and taken < steps:
            unfinished = True
            break
    gaps, held_share = fit_gap(snapshots, universe.fell_into)
    for sample, gap in zip(samples, gaps):
        sample["gap"] = gap
    record = {"overrides": overrides, "seed": seed, "cells": cells, "steps": steps,
              "steps_done": samples[-1]["step"], "unfinished": unfinished, "ended": ended,
              "held_share": held_share, "seconds": time.time() - started,
              "series": {name: [sample[name] for sample in samples] for name in samples[0]}}
    if unfinished or len(samples) < 4:
        return {**record, "score": 0.0, "E": 0.0, "S": 0.0, "hold": 0.0,
                "error": f"unfinished: {samples[-1]['step']} of {steps} steps in {seconds} s"
                if unfinished else f"ended after {len(samples)} samples: {ended}"}
    counter = fit_score([{**sample, "tau": sample["time"]} for sample in samples], held_share)
    return {**record, **fit_score(samples, held_share),
            "counter": {name: counter[name] for name in ("score", "E", "dip", "gap_dip")}}
