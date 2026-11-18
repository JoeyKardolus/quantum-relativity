"""Ranges: the engine settings the fit searches, and how far.

Contract: constants only, fixed on 2026-11-18 before any search on the rebuilt engine;
every default of `physics.engine.Settings` lies inside its range.
  LOG_RANGES     searched evenly in the logarithm:
    pull_strength  1e-9 .. 1e-4     light_speed    0.1 .. 4 (the pull's speed)
    push_strength  2e-9 .. 2e3      squeeze_rate   0.01 .. 100
    collapse_thin  1.2 .. 30        tear_length    11 .. 1e5
    clock_packing  64.8 .. 6.48e5   friction       0.01 .. 1
    volume_quanta  1000 .. 10000    joint_quanta   30 .. 300 (the start's quanta a box)
  LINEAR_RANGES  searched evenly: the start's unevenness of the volumes 0 .. 0.9 and of
                 the joints 0 .. 0.9
  SEARCHED       the twelve names, the log ones first: the axes of the unit cube
Not searched: `step_time` (the counter's step, no rule), `size_constant` (it sets the
unit of length), `slope` and `inner_unevenness` (the shape of the start's unevenness)
and the split's speed range (`slowest_speed`, `fastest_speed`). No dark-energy,
base-push, seal or crossing-rate setting exists to search.
"""
from __future__ import annotations

LOG_RANGES = {"pull_strength": (1e-9, 1e-4), "light_speed": (0.1, 4.0), "push_strength": (2e-9, 2e3),
              "squeeze_rate": (0.01, 100.0), "collapse_thin": (1.2, 30.0), "tear_length": (11.0, 1e5),
              "clock_packing": (64.8, 6.48e5), "friction": (0.01, 1.0), "volume_quanta": (1000.0, 10000.0),
              "joint_quanta": (30.0, 300.0)}
LINEAR_RANGES = {"unevenness": (0.0, 0.9), "joint_unevenness": (0.0, 0.9)}
SEARCHED = list(LOG_RANGES) + list(LINEAR_RANGES)
