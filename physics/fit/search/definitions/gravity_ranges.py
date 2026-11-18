"""Gravity ranges: the settings phase 1 of the fit searches, and how far.

Contract: constants and one function, fixed on 2026-11-18 before any phase 1 search;
every default of `physics.engine.Settings` lies inside its range. Phase 1 fits only for
gravity (bible, "The fit goes in phases"), so only the settings that matter for it are
searched, all evenly in the logarithm:
  pull_strength  1e-9 .. 1e-5     how hard energy pulls
  push_strength  2e-8 .. 2e-2     how hard a joint pushes: with the balance, how thick it starts
  friction       1e-4 .. 0.5      the share of crossing energy a joint takes the direction off
  light_speed    1 .. 40          how fast the pull follows the energy
  clock_packing  6480 .. 6.48e15  the packing at which a clock runs at half rate
  collapse_thin  3 .. 300         how far a joint is squeezed before it collapses
  tear_length    20 .. 2e4        how long an emptied volume gets before it tears
`GRAVITY_SEARCHED` is the seven names in that order. `gravity_point(unit)` maps one
coordinate between 0 and 1 per name to `{name: value}`; ValueError for another count.
Not searched in phase 1: the start's amounts, `step_time`, `size_constant`,
`squeeze_rate` and the split's speeds. No dark-energy, base-push, seal or drift setting
exists to search.
"""
from __future__ import annotations

import math

GRAVITY_RANGES = {"pull_strength": (1e-9, 1e-5), "push_strength": (2e-8, 2e-2), "friction": (1e-4, 0.5),
                  "light_speed": (1.0, 40.0), "clock_packing": (6480.0, 6.48e15), "collapse_thin": (3.0, 300.0),
                  "tear_length": (20.0, 2e4)}
GRAVITY_SEARCHED = list(GRAVITY_RANGES)


def gravity_point(unit) -> dict:
    """The engine settings a point of the unit cube stands for, evenly in the logarithm."""
    if len(unit) != len(GRAVITY_SEARCHED):
        raise ValueError(f"a point has {len(GRAVITY_SEARCHED)} coordinates, got {len(unit)}")
    return {name: float(math.exp(math.log(low) + float(coordinate) * (math.log(high) - math.log(low))))
            for (name, (low, high)), coordinate in zip(GRAVITY_RANGES.items(), unit)}
