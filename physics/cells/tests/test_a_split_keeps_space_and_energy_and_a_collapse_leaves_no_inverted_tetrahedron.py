"""Topology changes only by its two events, both leaving only tetrahedra: a volume split is the 1-4 move and keeps
space and energy exactly; a joint collapse draws an edge into one corner, keeps energy exactly and inverts nothing."""
from dataclasses import replace

import numpy as np

from physics.cells import (EngineSettings, cell_volumes, collapse_edge, engine_stats, joint_rest,
                           new_engine, split_cell, step, tet_faces)


def test_the_1_4_move_keeps_every_other_tetrahedron_and_the_total_space():
    engine = new_engine(200, 6, 0.5)
    before = cell_volumes(engine.corners, engine.tets)
    corners, tets, children, shares = split_cell(engine.corners, engine.tets, 7)
    after = cell_volumes(corners, tets)
    assert len(tets) == len(engine.tets) + 3 and len(corners) == len(engine.corners) + 1
    assert (after > 0).all()
    others = np.delete(np.arange(len(engine.tets)), 7)
    assert np.array_equal(after[others], before[others])
    assert np.isclose(after[children].sum(), before[7], rtol=1e-13)
    assert np.isclose(after.sum(), before.sum(), rtol=1e-13)
    assert np.allclose(shares, 0.25, rtol=1e-12)


def test_one_thin_volume_splits_into_four_with_six_new_joints_and_the_same_energy():
    engine = new_engine(200, 6, 0.5)
    grid = engine.grid
    density = np.sort(grid.energy / grid.volume)
    settings = EngineSettings.all_on(leak=0.0, push_limit=0.0,
                                     split_threshold=float(density[:2].mean()))
    parent = int(np.argmin(grid.energy / grid.volume))
    after = step(engine, settings)
    assert after.splits == 1 and after.grid.cell_count == grid.cell_count + 3
    assert len(after.grid.joints) == len(grid.joints) + 6
    children = [parent, grid.cell_count, grid.cell_count + 1, grid.cell_count + 2]
    assert np.isclose(after.grid.energy[children].sum(), grid.energy[parent], rtol=1e-14)
    assert np.isclose(after.grid.quanta[children].sum(), grid.quanta[parent], rtol=1e-14)
    assert np.isclose(after.grid.total_energy(), grid.total_energy(), rtol=1e-14)
    assert (cell_volumes(after.corners, after.tets) > 0).all()
    assert engine_stats(after)["neighbour_changes"] == 0


def test_an_inner_edge_collapse_keeps_its_space_and_every_total_and_inverts_nothing():
    engine = new_engine(300, 3, 0.5)
    faces, face_cells = tet_faces(engine.tets)
    outer = np.unique(faces[face_cells[:, 1] < 0])
    inner = np.setdiff1d(np.arange(len(engine.corners)), outer)
    values = np.column_stack([engine.grid.energy, engine.grid.quanta])
    result = None
    for corner in inner:
        partner = next(other for other in engine.tets[(engine.tets == corner).any(axis=1)].ravel()
                       if other != corner and other in inner)
        result = collapse_edge(engine.corners, engine.tets, values, (corner, partner))
        if result is not None:
            break
    assert result is not None
    corners, tets, handed, kept, gone = result
    star = (engine.tets == corner).any(axis=1) | (engine.tets == partner).any(axis=1)
    space_before = cell_volumes(engine.corners, engine.tets[star]).sum()
    space_after = cell_volumes(corners, tets[star[kept]]).sum()
    assert np.isclose(space_after, space_before, rtol=1e-12)
    assert len(tets) < len(engine.tets) and not (tets == gone).any()
    assert (cell_volumes(corners, tets) > 0).all()
    assert np.allclose(handed.sum(axis=0), values.sum(axis=0), rtol=1e-14)


def test_one_joint_below_its_limit_collapses_and_the_energy_is_kept_exactly():
    """The joint is set at a tenth of rest with the push off, so it stays there for the step."""
    engine = new_engine(300, 6, 0.5)
    grid = engine.grid
    settings = EngineSettings.all_on(leak=0.0, split_threshold=0.0, joint_push=0.0)
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    inner_corner = np.ones(len(engine.corners), dtype=bool)
    inner_corner[engine.faces[engine.face_cells[:, 1] < 0].ravel()] = False
    joint_face = engine.face_joint >= 0
    deep = np.flatnonzero(joint_face & inner_corner[engine.faces].all(axis=1))
    chosen = int(engine.face_joint[deep[0]])
    thickness = grid.thickness.copy()
    thickness[chosen] = 0.1 * rest[chosen]
    rate = grid.rate.copy()
    rate[chosen] = 0.0
    after = step(replace(engine, grid=replace(grid, thickness=thickness, rate=rate)), settings)
    assert after.collapses == 1 and after.grid.cell_count < grid.cell_count
    assert np.isclose(after.grid.total_energy(), grid.total_energy(), rtol=1e-14)
    assert np.isclose(after.grid.quanta.sum(), grid.quanta.sum(), rtol=1e-14)
    assert (cell_volumes(after.corners, after.tets) > 0).all()
    assert (np.bincount(after.face_cells[after.face_cells >= 0], minlength=after.grid.cell_count) == 4).all()
