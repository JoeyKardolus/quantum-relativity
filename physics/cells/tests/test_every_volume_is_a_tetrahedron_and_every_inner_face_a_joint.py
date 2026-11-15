"""The mesh is the geometry: every energy volume is a tetrahedron with exactly 4 faces, every inner face is one joint."""
from dataclasses import replace

import numpy as np

from physics.cells import (EngineSettings, cell_volumes, face_joints, first_mesh, frame_arrays,
                           mesh_message, new_engine, step, tet_faces)


def _faces_per_cell(face_cells, count):
    real = face_cells[face_cells >= 0]
    return np.bincount(real, minlength=count)


def test_the_first_mesh_is_positive_tetrahedra_filling_their_space():
    corners, tets = first_mesh(500, np.random.default_rng(1))
    volumes = cell_volumes(corners, tets)
    assert tets.shape[1] == 4 and (volumes > 0).all()
    assert 0.8 * 500 < len(tets) < 1.2 * 500
    assert np.array_equal(np.unique(tets), np.arange(len(corners)))


def test_every_cell_has_four_faces_and_every_inner_face_is_one_joint_through_events():
    """Pull at three times its default for collapses, then a split size just below the biggest volumes, so a few
    split each step."""
    engine = new_engine(300, 2, 0.5)
    pulled = EngineSettings.all_on(split_size=np.inf, attraction=0.015)
    for index in range(43):
        biggest = float(np.sort(engine.grid.volume)[-3:].mean()) / engine.start_mean_volume
        settings = pulled if index < 40 else replace(pulled, split_size=biggest)
        engine = step(engine, settings)
        grid = engine.grid
        assert (_faces_per_cell(engine.face_cells, grid.cell_count) == 4).all()
        inner = (engine.face_cells >= 0).all(axis=1)
        assert (engine.face_joint[inner] >= 0).all() and (engine.face_joint[~inner] == -1).all()
        assert np.array_equal(np.sort(engine.face_joint[inner]), np.arange(len(grid.joints)))
        assert np.array_equal(np.sort(grid.joints[engine.face_joint[inner]], axis=1),
                              np.sort(engine.face_cells[inner], axis=1))
        faces, face_cells = tet_faces(engine.tets)
        assert np.array_equal(faces, engine.faces) and np.array_equal(face_cells, engine.face_cells)
        assert np.array_equal(face_joints(face_cells)[0], grid.joints)
        assert (cell_volumes(engine.corners, engine.tets) > 0).all()
    assert engine.splits > 0 and engine.collapses > 0


def test_the_page_gets_the_topology_once_and_small_binary_frames():
    engine = new_engine(200, 1, 0.5)
    mesh = mesh_message(engine)
    corners, pull, push = frame_arrays(engine)
    assert mesh["corner_count"] == len(corners) and corners.dtype == np.float32
    assert len(mesh["faces"]) == len(mesh["face_cells"]) == len(mesh["face_joint"])
    assert all(len(face) == 3 for face in mesh["faces"])
    assert pull.dtype == np.uint8 and len(pull) == mesh["cells"]
    assert push.dtype == np.uint8 and len(push) == mesh["joints"]
