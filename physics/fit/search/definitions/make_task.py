"""Make task: one run for the search to do.

Contract: `make_task(phase, index, unit, seed)` returns
  id         "<phase>:<index>:<seed>": a run whose id is already in the results file is
             never run again, so a stopped search resumes
  phase      "defaults", "random", "refine<r>", "hold" or "sweep"
  seed       the universe's seed
  unit       the unit-cube point as plain floats, or None for the default settings
  overrides  the engine settings that point stands for ({} for the defaults)
"""
from __future__ import annotations

from physics.fit.search.definitions.unit_to_overrides import unit_to_overrides


def make_task(phase: str, index, unit, seed: int) -> dict:
    """One run to do: its id, phase, unit point, overrides and seed."""
    point = None if unit is None else [float(value) for value in unit]
    return {"id": f"{phase}:{index}:{seed}", "phase": phase, "seed": seed, "unit": point,
            "overrides": {} if point is None else unit_to_overrides(point)}
