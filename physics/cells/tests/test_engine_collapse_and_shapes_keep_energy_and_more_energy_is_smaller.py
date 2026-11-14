"""Engine joint collapse keeps energy and lowers the cell count; the shapes keep energy;
an energy volume with more energy ends up smaller.

  collapse  a step with push_limit 0.99 (joints give way almost at once) removes
            seeds: fewer cells, engine_stats counts them, every remaining cell has a
            shape, the joints are still exactly the shared faces, and the total
            energy is what the rules left before collapse (move_space), to rounding.
  energy    the leak (any seam openness), joint collapse and the engine's shape part (seeds, weights,
            box, re-tessellation, reconciling joints) keep the total energy and total
            space: a full step ends with exactly what apply_rules gave, and the box
            holds exactly that space.
  smaller   after new_engine has settled the shapes (500 volumes, seed 6,
            unevenness 0.3), tessellated volume falls with energy: correlation
            below -0.8, and the most energetic quarter is smaller on average than
            the least energetic quarter.
"""
from __future__ import annotations

import numpy as np

from physics.cells import (EngineSettings, apply_rules, engine_stats, joint_collapse, leak,
                           move_space, new_engine, reconcile_joints, step)


def test_collapse_keeps_energy_and_lowers_the_cell_count():
    engine = new_engine(500, seed=2, unevenness=0.1)
    settings = EngineSettings.all_on(push_limit=0.99)
    before_collapse = move_space(engine.grid, settings).total_energy()
    stepped = step(engine, settings)
    fell_in = engine.grid.cell_count - stepped.grid.cell_count
    assert fell_in > 0
    assert engine_stats(stepped)["collapses"] == fell_in
    assert np.isclose(stepped.grid.total_energy(), before_collapse, rtol=1e-12)
    assert len(stepped.positions) == stepped.grid.cell_count
    assert (stepped.tessellation.volume > 0).all()
    assert np.array_equal(stepped.grid.joints, stepped.tessellation.pairs)
    assert np.isclose(stepped.tessellation.volume.sum(), stepped.side ** 3, rtol=1e-9)


def test_leak_collapse_and_shapes_keep_energy():
    engine = new_engine(500, seed=3, unevenness=0.2)
    grid = engine.grid
    pull = np.random.default_rng(3).uniform(0.1, 2.0, grid.cell_count)
    openness = np.random.default_rng(4).uniform(0.0, 1.0, len(grid.seams))
    assert np.isclose(leak(grid.joints, grid.seams, grid.energy, pull, openness, 0.2).sum(),
                      grid.total_energy(), rtol=1e-13)
    giving_way = np.zeros(len(grid.joints), dtype=bool)
    giving_way[::50] = True
    assert np.isclose(joint_collapse(grid, giving_way).total_energy(), grid.total_energy(),
                      rtol=1e-13)
    shuffled = reconcile_joints(grid, grid.joints[::2], grid.seams, EngineSettings.all_on())
    assert np.array_equal(shuffled.energy, grid.energy)
    assert np.isclose(shuffled.total_space(), grid.total_space(), rtol=1e-12)
    settings = EngineSettings.all_on()
    for _ in range(5):
        ruled = apply_rules(engine.grid, settings)
        engine = step(engine, settings)
        assert np.isclose(engine.grid.total_energy(), ruled.total_energy(), rtol=1e-13)
        assert np.isclose(engine.grid.total_space(), ruled.total_space(), rtol=1e-12)
        assert np.isclose(engine.side ** 3, engine.grid.total_space(), rtol=1e-12)


def test_more_energy_ends_up_smaller():
    engine = new_engine(500, seed=6, unevenness=0.3)
    energy, volume = engine.grid.energy, engine.tessellation.volume
    assert np.corrcoef(energy, volume)[0, 1] < -0.8
    low, high = np.quantile(energy, [0.25, 0.75])
    assert volume[energy >= high].mean() < volume[energy <= low].mean()
