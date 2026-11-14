"""The rules and the engine are reproducible: no randomness, so the same start and settings give identical results."""
from __future__ import annotations

import numpy as np

from physics.cells import EngineSettings, apply_rules, new_engine, step


def _run(steps):
    grid = new_engine(400, seed=5, unevenness=0.1).grid
    settings = EngineSettings.all_on()
    for _ in range(steps):
        grid = apply_rules(grid, settings)
    return grid


def test_the_rules_are_reproducible():
    first, second = _run(10), _run(10)
    assert first.cell_count == second.cell_count
    for name in ("joints", "energy", "volume", "joint_volume", "squeeze", "charge",
                 "merged_into"):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_the_engine_is_reproducible():
    runs = []
    for _ in range(2):
        engine = new_engine(300, seed=5, unevenness=0.1)
        for _ in range(5):
            engine = step(engine, EngineSettings.all_on())
        runs.append(engine)
    first, second = runs
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.grid.joints, second.grid.joints)
    assert np.array_equal(first.grid.energy, second.grid.energy)
    assert np.array_equal(first.tessellation.corners, second.tessellation.corners)
