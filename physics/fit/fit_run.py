"""fit_run: one fit-set universe, stepped and scored on expansion shape and structure.

Contract: fit_run(overrides, seed, cells, boxes, steps, every) -> dict. Builds
new_universe(cells, boxes, seed), steps it with Settings.all_on(**overrides) and samples every
`every` steps. The baseline is the first sample after step 0 (the first step's size correction
is a jump, not the dynamics). score = 0.5 E + 0.5 S:
  E  expansion dip min(r1 - r2, r3 - r2) / mean |r| over the occupied radius's growth rate in
     thirds of the run, clipped to +-1; -1 when space does not grow or shrinks; minus 1 when
     edge_share > 0.01 (the walls are reached, no longer open space). E > 0 only when growth
     slows, then speeds up.
  S  mean of tanh(ln contrast ratio), tanh(lump gain / 5) and tanh(large-scale minus
     small-scale ln power growth), on the central cube the start fills.
Only the engine's public face and the scenes' shared measures, as the scenes use them.
"""
from __future__ import annotations

import math
import time

import numpy as np

from physics.potts import Settings, new_universe, step, stats
from physics.scenes import box_energies, growth_rates, power_spectrum


def _central_density(universe, lower, upper):
    a_energy, b_energy = box_energies(universe)
    density = a_energy + b_energy
    return density[lower:upper, lower:upper, lower:upper]


def _measure(universe, settings, lower, upper):
    now = stats(universe, settings)
    density = _central_density(universe, lower, upper)
    wavenumbers, power = power_spectrum(density)
    contrast = float(np.sqrt(np.mean((density / density.mean() - 1.0) ** 2)))
    return {"occupied_radius": (3.0 * now["occupied"] / (4.0 * math.pi)) ** (1.0 / 3.0),
            "occupied_growth": now["occupied_growth"], "edge_share": now["edge_share"],
            "lumps": now["lumps"], "collapses": now["collapses"], "contrast": contrast,
            "wavenumbers": wavenumbers.tolist(), "power": power.tolist()}


def _expansion_score(times, radii, final):
    _, rates = growth_rates(times, radii)
    thirds = np.array_split(rates, 3)
    r1, r2, r3 = (float(np.mean(part)) for part in thirds)
    scale = (abs(r1) + abs(r2) + abs(r3)) / 3.0 + 1e-9
    dip = min(r1 - r2, r3 - r2) / scale
    score = float(np.clip(dip, -1.0, 1.0))
    if final["occupied_growth"] < 1.02 or min(r1, r2, r3) <= 0.0:
        score = -1.0
    if final["edge_share"] > 0.01:
        score -= 1.0
    return score, {"r1": r1, "r2": r2, "r3": r3, "dip": dip}


def _structure_score(first, last):
    contrast_part = math.tanh(math.log(last["contrast"] / first["contrast"]))
    lump_part = math.tanh((last["lumps"] - first["lumps"]) / 5.0)
    growth = np.log(np.asarray(last["power"]) / np.asarray(first["power"]))
    wavenumbers = np.asarray(first["wavenumbers"])
    large = growth[wavenumbers <= 2].mean()
    small = growth[wavenumbers >= wavenumbers.max() / 2].mean()
    tilt_part = math.tanh(float(large - small))
    return (contrast_part + lump_part + tilt_part) / 3.0, {
        "Sc": contrast_part, "Sl": lump_part, "St": tilt_part, "ln_growth_large": float(large),
        "ln_growth_small": float(small)}


def fit_run(overrides, seed, cells, boxes, steps, every):
    """Run one universe with Settings.all_on(**overrides) and return its scores and series."""
    started = time.time()
    settings = Settings.all_on(**overrides)
    universe = new_universe(cells, boxes, seed)
    owner = universe.owner.numpy()
    occupied_axis = np.nonzero((owner >= 0).any(axis=(1, 2)))[0]
    lower, upper = int(occupied_axis.min()), int(occupied_axis.max()) + 1
    samples = [(0.0, _measure(universe, settings, lower, upper))]
    for taken in range(1, steps + 1):
        step(universe, settings)
        if taken % every == 0 or taken == steps:
            samples.append((taken * settings.step_time, _measure(universe, settings, lower, upper)))
    times = [when for when, _ in samples]
    radii = [measured["occupied_radius"] for _, measured in samples]
    first, last = samples[1][1], samples[-1][1]
    expansion_score, expansion_parts = _expansion_score(times[1:], radii[1:], last)
    structure_score, structure_parts = _structure_score(first, last)
    return {"overrides": overrides, "seed": seed, "cells": cells, "boxes": boxes, "steps": steps,
            "score": 0.5 * expansion_score + 0.5 * structure_score, "E": expansion_score,
            "S": structure_score, **expansion_parts, **structure_parts,
            "occupied_growth_end": last["occupied_growth"], "edge_share_end": last["edge_share"],
            "lumps": [measured["lumps"] for _, measured in samples],
            "contrast": [measured["contrast"] for _, measured in samples],
            "radius": radii, "collapses_end": last["collapses"], "substeps": universe.substeps,
            "seconds": time.time() - started}
