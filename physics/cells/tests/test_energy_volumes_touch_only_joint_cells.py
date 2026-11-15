"""Every joint is a cell of its own: energy volumes have their own corners and touch only joint cells; joint cells
touch their two energy volumes by their big sides and each other only along their thin edges (rule 2)."""
from dataclasses import replace

import numpy as np

from physics.cells import (EngineSettings, cell_sizes, cell_volumes, face_joints, first_mesh, frame_arrays,
                           mesh_message, new_engine, step, tet_faces)
from physics.cells.cell_complex import KIND_CORNER, KIND_JOINT, KIND_SEAM, KIND_VOLUME


def _check_touching(engine):
    """Every contact rule of rule 2, on one engine."""
    grid, kind = engine.grid, engine.cell_kind
    count, joint_count = grid.cell_count, len(grid.joints)
    first, second = engine.face_cells[:, 0], engine.face_cells[:, 1]
    first_kind = kind[first]
    second_kind = np.where(second >= 0, kind[np.maximum(second, 0)], -1)
    on_volume = (first_kind == KIND_VOLUME) | (second_kind == KIND_VOLUME)
    other = np.where(first_kind == KIND_VOLUME, second_kind, first_kind)
    assert set(np.unique(other[on_volume]).tolist()) <= {KIND_JOINT, -1}
    assert (np.bincount(first[first_kind == KIND_VOLUME], minlength=count)[:count] == 4).all()
    assert not ((first_kind == KIND_JOINT) & (second_kind == KIND_JOINT)).any()
    assert not ((first_kind == KIND_JOINT) & (second_kind == -1)).any()
    big = (first_kind == KIND_VOLUME) & (second_kind == KIND_JOINT)
    pairs = np.unique(np.column_stack([second[big] - count, first[big]]), axis=0)
    assert np.array_equal(np.bincount(pairs[:, 0], minlength=joint_count), np.full(joint_count, 2))
    assert np.array_equal(pairs[:, 1].reshape(-1, 2), grid.joints)
    assert np.array_equal(engine.face_joint[big], second[big] - count)
    own = engine.tets.ravel()
    assert len(np.unique(own)) == len(own)
    assert np.array_equal(face_joints(tet_faces(engine.skeleton_tets)[1])[0], grid.joints)
    assert (np.bincount(kind, minlength=4)[[KIND_SEAM, KIND_CORNER]] > 0).all()


def test_the_first_mesh_is_positive_tetrahedra_filling_their_space():
    corners, tets = first_mesh(500, np.random.default_rng(1))
    volumes = cell_volumes(corners, tets)
    assert tets.shape[1] == 4 and (volumes > 0).all()
    assert 0.8 * 500 < len(tets) < 1.2 * 500
    assert np.array_equal(np.unique(tets), np.arange(len(corners)))


def test_energy_volumes_touch_only_joint_cells_through_collapses_and_splits():
    """Pull at three times its default for collapses, then a split size just below the biggest volumes, so a few
    split each step."""
    engine = new_engine(300, 2, 0.5)
    _check_touching(engine)
    pulled = EngineSettings.all_on(split_size=np.inf, attraction=0.015)
    for index in range(43):
        biggest = float(np.sort(engine.grid.volume)[-3:].mean()) / engine.start_mean_volume
        settings = pulled if index < 40 else replace(pulled, split_size=biggest)
        engine = step(engine, settings)
        _check_touching(engine)
        assert (cell_sizes(engine.corners, engine.faces, engine.face_cells, len(engine.cell_kind)) > 0).all()
    assert engine.splits > 0 and engine.collapses > 0


def test_the_page_gets_the_topology_and_each_cells_kind_once_and_small_binary_frames():
    engine = new_engine(200, 1, 0.5)
    mesh = mesh_message(engine)
    corners, pull, push = frame_arrays(engine)
    assert mesh["corner_count"] == len(corners) and corners.dtype == np.float32
    assert len(mesh["faces"]) == len(mesh["face_cells"]) == len(mesh["face_joint"])
    assert all(len(face) == 3 for face in mesh["faces"])
    assert pull.dtype == np.uint8 and len(pull) == mesh["cells"] == len(mesh["cell_kind"])
    assert push.dtype == np.uint8 and len(push) == mesh["joints"]
    assert mesh["cell_kind"][:mesh["energy_volumes"]] == [0] * mesh["energy_volumes"]
    assert (pull[mesh["energy_volumes"]:] == 0).all()
