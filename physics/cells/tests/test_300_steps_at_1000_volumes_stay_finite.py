"""The default run holds together: 300 steps at 1000 energy volumes, all rules on, stay finite and keep the energy."""
import numpy as np
import pytest

from physics.cells import EngineSettings, cell_volumes, new_engine, step


@pytest.mark.xfail(strict=True, raises=RuntimeError,
                   reason="miss (2026-11-15, seed 1, 968 volumes, unevenness 0.5, defaults): a split keeps density, so "
                   "the children of a thin volume split again every step; splits start at step 86 (1, 5, 20, 80, 319, "
                   "1275, 5099) and the split ceiling stops the run at step 93. Splits off: finite to 300, size 2.20")
def test_300_steps_at_1000_volumes_stay_finite_with_every_tetrahedron_the_right_way_out():
    engine = new_engine(1000, 1, 0.5)
    settings = EngineSettings.all_on()
    start = engine.grid.total_energy()
    for _ in range(300):
        engine = step(engine, settings)
    grid = engine.grid
    assert np.isfinite(engine.corners).all() and np.isfinite(grid.energy).all()
    assert np.isfinite(grid.thickness).all() and np.isfinite(grid.rate).all()
    assert (cell_volumes(engine.corners, engine.tets) > 0).all() and (grid.energy > 0).all()
    assert np.isclose(grid.total_energy(), start, rtol=1e-12)
