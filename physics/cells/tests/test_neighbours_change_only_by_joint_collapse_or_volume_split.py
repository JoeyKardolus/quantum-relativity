"""Corners only move: with no collapse and no split, no two energy volumes start or stop neighbouring."""
import numpy as np

from physics.cells import EngineSettings, cell_volumes, engine_stats, new_engine, step


def test_200_steps_without_collapse_or_split_change_no_neighbour_pair():
    engine = new_engine(200, 4, 0.5)
    settings = EngineSettings.all_on(push_limit=0.0, split_threshold=0.0)
    start_joints, start_cells = engine.grid.joints.copy(), engine.face_cells.copy()
    start_corners = engine.corners.copy()
    for _ in range(200):
        engine = step(engine, settings)
    assert engine.collapses == 0 and engine.splits == 0
    assert np.array_equal(engine.grid.joints, start_joints)
    assert np.array_equal(engine.face_cells, start_cells)
    assert engine_stats(engine)["neighbour_changes"] == 0
    assert not np.allclose(engine.corners, start_corners)
    assert (cell_volumes(engine.corners, engine.tets) > 0).all()
