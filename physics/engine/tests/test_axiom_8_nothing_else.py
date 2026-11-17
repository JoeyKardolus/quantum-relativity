"""Axiom 8, Nothing else: no rule beyond the eight acts, and one seed gives one run.

  - Settings holds exactly the strengths of the rules built, and none whose name says
    drift, spread, dark, base, charge, spring, ball or rest
  - no engine file is left from the machinery that treated Planck boxes as space
    (boxes changing cell, cells claiming boxes, joints swinging as springs)
  - a universe stepped twice from the same seed ends bit for bit the same, through
    tears and collapses
  - an id that is no cell does nothing: its boxes stay empty (checked at every step of
    both runs)
"""
import dataclasses
import pathlib

import numpy as np

from physics.engine import Settings, new_universe, step
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.axiom_run import axiom_run

_RULES = {"step_time", "light_speed", "pull_strength", "push_strength", "squeeze_rate", "seal_thin", "collapse_thin",
          "crossing_rate", "tear_length", "clock_packing", "fluid_pressure"}
_NEVER = ("drift", "spread", "dark", "base", "charge", "spring", "ball", "rest")
_GONE = ("exchange", "mortar_shift", "settle_counts", "cell_room", "joint_swing", "swing_table", "speed_handover",
         "wall_press", "window_read", "window_write", "cells_span", "host_boxes", "volume_split", "seams")


def test_settings_hold_only_the_rules_that_are_built():
    names = {field.name for field in dataclasses.fields(Settings)}
    hidden = sorted(name for name in names if any(word in name for word in _NEVER))
    print(f"settings: {sorted(names)}; names that say drift, spread, dark, base, charge, spring, ball or rest: {hidden}")
    assert names == _RULES and not hidden


def test_the_box_ownership_machinery_is_gone():
    here = pathlib.Path(__file__).resolve().parent.parent
    left = sorted(name for name in _GONE if (here / f"{name}.py").exists())
    print(f"engine files left from boxes-as-space: {left}")
    assert not left


@needs_gpu
def test_one_seed_gives_one_run_through_tears_and_collapses():
    ends = []
    for _ in range(2):
        state, settings = new_universe(16, unevenness=0.3), Settings(pull_strength=1e-6)
        for _ in range(12):
            step(state, settings)
        ends.append((state.quanta.numpy().copy(), state.phase.numpy().copy(), state.host("pull").copy(),
                     state.tears, state.collapses))
    print(f"16 places a side, uneven start, ten times the pull, 12 steps, twice: tears {ends[0][3]}, collapses "
          f"{ends[0][4]}")
    assert ends[0][3] == ends[1][3] > 0 and ends[0][4] == ends[1][4] > 0
    assert all(np.array_equal(first, second) for first, second in zip(ends[0][:3], ends[1][:3]))


@needs_gpu
def test_an_id_that_is_no_cell_does_nothing():
    for name in ("default", "eventful"):
        run = axiom_run(name)
        state = run["state"]
        dead = np.flatnonzero(state.kind_h[:state.count] == 0)
        left = sum(int(np.abs(state.block(cell)[0]).sum()) for cell in dead[:50])
        print(f"{name}: ids that stopped being a cell {len(dead)}; quanta left in the boxes of the first 50: {left}; "
              f"boxes that broke their cell over all steps: {sum(row['wrong_boxes'] for row in run['steps'])}")
        assert left == 0 and sum(row["wrong_boxes"] for row in run["steps"]) == 0
