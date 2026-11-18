"""Run universe: one fit-set universe, stepped, measured inside and scored.

Contract: `run_universe(overrides, seed, cells, steps, every, seconds=None, device="cuda")`
returns a dict. It builds `physics.engine` `new_universe(Settings(**overrides), cells,
seed)`: every override is a Settings field (the start's energies too), the universe wraps
and nothing is placed. It samples `measure_cells` and `take_snapshot` at the start and
every `every` steps.
Time is the observer's (time is local): "tau" adds up `observer_time` from sample to
sample. The counter-time score is returned beside it (`counter`) so the clock's part
can be seen.
A run is ended early, between steps, when the universe is over (`why_ended`; `ended`
names the mark). Its score is the same formula on the samples so far. A universe whose
energy volumes hold nothing at a sample (`NOTHING_LEFT`) cannot be measured: it is ended
without that sample and scores 0.
Returns the overrides, sizes, every `score_samples` part, the series (one list per
measured name, plus "step", "time", "tau", "gap"), `held_share`, wall `seconds`, and
`steps_done`, and `unfinished`: True when `seconds` of wall time ran out before `steps`.
An unfinished run, one with nothing left, and one ended with fewer samples than the score
needs, score 0 and say so in "error".
A name that is no Settings field raises TypeError; the engine's errors propagate.
"""
from __future__ import annotations

import time

from physics.engine import ENERGY_VOLUME, Settings, new_universe, step
from physics.fit.measures.definitions.observer_time import observer_time
from physics.fit.measures.logic.gap_between_lumps import gap_between_lumps
from physics.fit.measures.logic.held_share import held_share
from physics.fit.measures.logic.measure_cells import measure_cells
from physics.fit.measures.logic.read_universe import read_universe
from physics.fit.measures.logic.take_snapshot import take_snapshot
from physics.fit.score.definitions.score_marks import MIN_SAMPLES
from physics.fit.score.logic.score_samples import score_samples
from physics.fit.search.definitions.torn_out import torn_out
from physics.fit.search.definitions.why_ended import NOTHING_LEFT, why_ended

_NO_SCORE = {"score": 0.0, "E": 0.0, "S": 0.0, "hold": 0.0}


def _sample(before: dict | None, taken: int, settings: Settings, cells: dict) -> dict:
    """One sample: the step, the counter time, the observer's time so far and the measures."""
    measured = measure_cells(cells, settings.tear_length)
    tau = 0.0
    if before is not None:
        passed = (taken - before["step"]) * settings.step_time
        tau = before["tau"] + observer_time(passed, before["observer_rate"], measured["observer_rate"])
    return {"step": taken, "time": taken * settings.step_time, "tau": tau, **measured}


def _holds_energy(cells: dict) -> bool:
    """Whether any energy volume holds a quantum."""
    return bool(cells["held"][cells["kind"] == ENERGY_VOLUME].sum() > 0)


def _scored(samples: list, held: float, unfinished: bool, ended: str, done: int, steps: int, seconds) -> dict:
    """The score's parts for a run's samples, or a score of 0 with the reason it could not be scored."""
    if unfinished:
        return {**_NO_SCORE, "error": f"unfinished: {done} of {steps} steps in {seconds} s"}
    if ended == NOTHING_LEFT:
        return {**_NO_SCORE, "error": f"ended: {ended}"}
    if len(samples) < MIN_SAMPLES:
        return {**_NO_SCORE, "error": f"ended after {len(samples)} samples: {ended}"}
    counter = score_samples([{**sample, "tau": sample["time"]} for sample in samples], held)
    return {**score_samples(samples, held),
            "counter": {name: counter[name] for name in ("score", "E", "dip", "gap_dip")}}


def run_universe(overrides: dict, seed: int, cells: int, steps: int, every: int, seconds: float | None = None,
                 device: str = "cuda") -> dict:
    """Run one wrapping universe from the overrides; return its score, parts and series."""
    started = time.time()
    settings = Settings(**overrides)
    universe = new_universe(settings, cells, seed, device=device)
    start_volumes = (universe.side // 2) ** 3
    read = read_universe(universe)
    samples, snapshots = [_sample(None, 0, settings, read)], [take_snapshot(read)]
    unfinished, ended, done = False, "", 0
    for taken in range(1, steps + 1):
        step(universe, settings)
        done = taken
        if taken % every == 0 or taken == steps or torn_out(universe.tears, start_volumes):
            read = read_universe(universe, read)
            if not _holds_energy(read):
                ended = NOTHING_LEFT
                break
            samples.append(_sample(samples[-1], taken, settings, read))
            snapshots.append(take_snapshot(read))
            ended = why_ended(universe.tears, start_volumes, samples[-1]["alive"], samples[-1]["spent_share"])
            if ended:
                break
        if seconds is not None and time.time() - started > seconds and taken < steps:
            unfinished = True
            break
    held = held_share(snapshots, universe.host("fell_into"))
    for sample, gap in zip(samples, gap_between_lumps(snapshots)):
        sample["gap"] = gap
    return {"overrides": overrides, "seed": seed, "cells": cells, "steps": steps,
            "steps_done": done, "unfinished": unfinished, "ended": ended, "held_share": held,
            "series": {name: [sample[name] for sample in samples] for name in samples[0]},
            **_scored(samples, held, unfinished, ended, done, steps, seconds), "seconds": time.time() - started}
