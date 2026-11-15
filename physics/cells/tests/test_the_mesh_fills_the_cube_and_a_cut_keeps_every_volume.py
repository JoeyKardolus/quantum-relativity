"""The mesh is the geometry: cells share corners and faces, fill the open cube, and a cut moves no other cell."""
import numpy as np

from physics.cells import (build_fan, cell_volumes, first_mesh, frame_arrays, mesh_message,
                           new_engine, split_cell)


def test_the_first_mesh_fills_the_cube_with_shared_faces():
    points = np.random.default_rng(1).random((150, 3)) * 5.0
    corners, faces, face_cells = first_mesh(points, 5.0, np.ones(150))
    volumes = cell_volumes(corners, build_fan(faces, face_cells, 150))
    assert (volumes > 0).all()
    assert np.isclose(volumes.sum(), 125.0, rtol=1e-12)
    pairs = np.sort(face_cells[(face_cells >= 0).all(axis=1)], axis=1)
    assert len(np.unique(pairs, axis=0)) == len(pairs)


def test_a_cut_on_flat_faces_keeps_every_other_cell_and_the_halves_sum_to_the_whole():
    points = np.random.default_rng(2).random((150, 3)) * 5.0
    corners, faces, face_cells = first_mesh(points, 5.0, np.ones(150))
    before = cell_volumes(corners, build_fan(faces, face_cells, 150))
    cut = split_cell(corners, faces, face_cells, np.zeros(len(faces), dtype=np.int64), 7, 150)
    assert cut is not None
    new_corners, new_faces, new_cells, _, share = cut
    after = cell_volumes(new_corners, build_fan(new_faces, new_cells, 151))
    others = np.delete(np.arange(150), 7)
    assert np.allclose(after[others], before[others], rtol=1e-12)
    assert np.isclose(after[7] + after[150], before[7], rtol=1e-12)
    assert np.isclose(share, after[7] / before[7], rtol=1e-12)


def test_the_page_gets_the_topology_once_and_small_binary_frames():
    engine = new_engine(200, 1, 0.5)
    mesh = mesh_message(engine)
    corners, pull, push = frame_arrays(engine)
    assert mesh["corner_count"] == len(corners) and corners.dtype == np.float32
    assert len(mesh["faces"]) == len(mesh["face_cells"]) == len(mesh["face_joint"])
    assert pull.dtype == np.uint8 and len(pull) == mesh["cells"]
    assert push.dtype == np.uint8 and len(push) == mesh["joints"]
