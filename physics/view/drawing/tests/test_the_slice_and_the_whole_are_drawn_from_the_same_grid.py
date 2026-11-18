"""Drawing as a whole: the whole grid and every slice of it are the same cells, the same colours, the same places."""
import numpy as np

from physics.view.drawing.definitions.camera_marks import CORNER
from physics.view.drawing.definitions.orbit_camera import orbit_camera
from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH
from physics.view.drawing.logic.far_to_near import far_to_near
from physics.view.drawing.logic.grid_triangles import grid_triangles
from physics.view.drawing.logic.view_fit import view_fit
from physics.view.tests.grid_cases import FakeSource

SIDE = FakeSource.SIDE


def test_the_slices_together_are_the_whole_grid_cell_for_cell():
    pieces = FakeSource().view()
    blocks, block_colours, _, drawn = grid_triangles(pieces, SHOW_BOTH, None)
    seen = 0
    for layer in range(SIDE):
        mine = pieces.place[:, 2] == layer
        faces, colours, _, count = grid_triangles(pieces, SHOW_BOTH, layer)
        seen += count
        assert np.array_equal(colours, block_colours[mine]), "a cell has one colour in both views"
        middle = 0.5 * (blocks[mine][:, :4] + blocks[mine][:, 4:])
        assert np.allclose(faces[:, [0, 1, 3, 2]], middle), "its face goes through the middle of its own block"
    assert seen == drawn == SIDE ** 3


def test_the_fitted_camera_sees_every_corner_of_the_grid_from_the_corner_view():
    pieces = FakeSource().view()
    centre, distance, _ = view_fit(pieces.low, pieces.high)
    matrix = orbit_camera(centre, distance, *CORNER, aspect=4 / 3)
    points = np.concatenate([pieces.corners.reshape(-1, 3), np.ones((len(pieces.cell) * 8, 1))], axis=1) @ matrix.T
    assert (points[:, 3] > 0).all(), "nothing behind the eye"
    assert np.abs(points[:, :2] / points[:, 3:]).max() < 1.0, "nothing outside the picture"
    order = far_to_near(pieces.corners.mean(axis=1), matrix)
    assert pieces.place[order[-1]].tolist() == [0, 0, 0], "the cell at lattice index 0 is the nearest, drawn last"
