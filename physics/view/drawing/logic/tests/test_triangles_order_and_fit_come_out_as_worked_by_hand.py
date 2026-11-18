"""grid_triangles, far_to_near and view_fit on plain arrays (no engine, no OpenGL)."""
import numpy as np
import pytest

from physics.view.colours.definitions.palette import JOINT_BLUE, VOLUME_RED
from physics.view.drawing.definitions.camera_marks import FACE_FILL, FILL
from physics.view.drawing.definitions.orbit_camera import orbit_camera
from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOW_JOINTS
from physics.view.drawing.logic.far_to_near import far_to_near
from physics.view.drawing.logic.grid_triangles import grid_triangles
from physics.view.drawing.logic.view_fit import view_fit
from physics.view.tests.grid_cases import FakeSource

SIDE = FakeSource.SIDE


def test_a_slice_is_one_layer_two_triangles_a_cell_and_kinds_can_be_shown_alone():
    pieces = FakeSource().view()
    corners, colours, pattern, drawn = grid_triangles(pieces, SHOW_BOTH, 4)
    assert drawn == SIDE * SIDE and corners.shape == (drawn, 4, 3) and colours.shape == (drawn, 4)
    assert corners.dtype == np.float32 and colours.dtype == np.uint8
    assert pattern.tolist() == [0, 1, 2, 0, 2, 3], "two triangles a cell"
    assert np.ptp(corners[:, :, 2], axis=1).max() < 3.0, "each cell's face is flat, in the middle of its block"
    _, reds, _, volumes = grid_triangles(pieces, SHOW_ENERGY_VOLUMES, 4)
    _, blues, _, joints = grid_triangles(pieces, SHOW_JOINTS, 4)
    assert volumes == 16 and joints == 48
    assert (reds[:, :3] == VOLUME_RED).all() and (blues[:, :3] == JOINT_BLUE).all()
    assert grid_triangles(pieces, SHOW_ENERGY_VOLUMES, 5)[3] == 0, "an odd layer holds only joints"
    with pytest.raises(ValueError):
        grid_triangles(pieces, 7, 4)


def test_the_whole_grid_draws_every_cell_as_a_block_so_the_inside_shows_through():
    pieces = FakeSource().view()
    corners, colours, pattern, drawn = grid_triangles(pieces, SHOW_BOTH, None)
    assert drawn == SIDE ** 3 and corners.shape == (drawn, 8, 3) and colours.shape == (drawn, 4)
    assert len(pattern) == 36 and sorted(set(pattern.tolist())) == list(range(8)), "six faces, twelve triangles"
    assert np.array_equal(corners, pieces.corners)
    assert grid_triangles(pieces, SHOW_ENERGY_VOLUMES, None)[3] == (SIDE // 2) ** 3
    assert grid_triangles(pieces, SHOW_JOINTS, None)[3] == SIDE ** 3 - (SIDE // 2) ** 3


def test_cells_are_ordered_far_to_near_from_wherever_the_camera_stands():
    centres = FakeSource().view().corners.mean(axis=1)
    middle = centres.mean(axis=0)
    for yaw, pitch in ((0.0, 0.0), (0.6 + np.pi, -0.35), (2.0, 1.2)):
        eye = middle + 400.0 * np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
        order = far_to_near(centres, orbit_camera(middle, 400.0, yaw, pitch, 1.5))
        assert order.dtype == np.uint32 and sorted(order.tolist()) == list(range(len(centres))), "every cell once"
        along = (centres[order] - eye) @ (middle - eye) / 400.0
        assert along[0] == along.max() and along[-1] == along.min()
        assert np.diff(along).max() < np.ptp(along) / 1000.0, "never a nearer cell before a farther one"
    assert len(far_to_near(np.zeros((0, 3)), orbit_camera(middle, 400.0, 0.0, 0.0, 1.5))) == 0


def test_three_cells_in_a_row_along_z_come_back_farthest_first():
    centres = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    assert far_to_near(centres, orbit_camera(np.zeros(3), 10.0, 0.0, 0.0, 1.0)).tolist() == [1, 2, 0]
    assert far_to_near(centres, orbit_camera(np.zeros(3), 10.0, np.pi, 0.0, 1.0)).tolist() == [0, 2, 1]


def test_the_view_is_fitted_to_the_frames_widest_side_and_closer_for_the_slice():
    centre, distance, extent = view_fit([-1.0, 0.0, 2.0], [3.0, 2.0, 12.0], 60.0)
    assert centre.tolist() == [1.0, 1.0, 7.0] and extent == 10.0 and np.isclose(distance, FILL * 10.0 / 0.5)
    assert np.isclose(view_fit([-1.0, 0.0, 2.0], [3.0, 2.0, 12.0], 60.0, FACE_FILL)[1], FACE_FILL * 10.0 / 0.5)
    assert view_fit([0, 0, 0], [0.1, 0.1, 0.1], 60.0)[2] == 1.0
    assert np.isclose(view_fit([0, 0, 0], [20, 20, 20], 60.0)[1], 2 * view_fit([0, 0, 0], [10, 10, 10], 60.0)[1]), \
        "a frame twice as wide is seen from twice as far: the grid stays still on screen"
