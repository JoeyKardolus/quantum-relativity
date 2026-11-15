"""The default run holds together: 300 steps at 1000 energy volumes, all rules on, stay finite and keep the energy."""
import numpy as np

from physics.cells import EngineSettings, cell_sizes, cell_volumes, new_engine, step


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
    assert (cell_sizes(engine.corners, engine.faces, engine.face_cells, len(engine.cell_kind)) > 0).all()
    assert np.isclose(grid.total_energy(), start, rtol=1e-12)
