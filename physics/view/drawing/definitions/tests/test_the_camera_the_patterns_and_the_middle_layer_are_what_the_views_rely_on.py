"""The drawing's definitions, on numbers worked out by hand."""
import numpy as np

from physics.view.drawing.definitions.block_faces import BLOCK_PATTERN, FACE_PATTERN, FACE_SHADE, MAX_PATTERN
from physics.view.drawing.definitions.camera_marks import CORNER, FACING, FOV_DEGREES, LEVELS
from physics.view.drawing.definitions.middle_layer import middle_layer
from physics.view.drawing.definitions.orbit_camera import orbit_camera
from physics.view.drawing.definitions.shader_text import FRAGMENT, VERTEX
from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOWN


def test_the_slice_starts_on_an_even_layer_where_energy_volumes_lie():
    assert [middle_layer(side) for side in (4, 6, 8, 16, 32, 34)] == [2, 2, 4, 8, 16, 16]


def test_a_blocks_twelve_triangles_are_two_on_each_of_its_six_faces():
    corners = np.array([[c & 1, c >> 1 & 1, c >> 2 & 1] for c in range(8)])
    triangles = corners[BLOCK_PATTERN].reshape(12, 3, 3)
    faces = []
    for triangle in triangles:
        flat = [axis for axis in range(3) if np.ptp(triangle[:, axis]) == 0]
        assert len(flat) == 1, "every triangle lies in one face of the block"
        faces.append((flat[0], int(triangle[0, flat[0]])))
        assert abs(np.linalg.det(np.delete(triangle[1:] - triangle[0], flat[0], axis=1))) == 1.0, "half a unit face"
    assert sorted(faces) == sorted([(axis, side) for axis in range(3) for side in (0, 1)] * 2)
    assert len(BLOCK_PATTERN) == MAX_PATTERN == 36 and FACE_PATTERN.tolist() == [0, 1, 2, 0, 2, 3]
    assert "pattern[36]" in VERTEX and "opacity_power" in FRAGMENT and "face_shade" in FRAGMENT
    assert len(FACE_SHADE) == 3 and len(set(FACE_SHADE)) == 3 and max(FACE_SHADE) == 1.0, "each axis its own shade"


def test_the_camera_puts_its_centre_in_the_middle_and_right_is_right_and_up_is_up():
    centre = np.array([3.0, -2.0, 5.0])
    matrix = orbit_camera(centre, 10.0, *FACING, aspect=2.0)

    def seen(point):
        clip = matrix @ np.append(point, 1.0)
        return clip[:2] / clip[3], clip[3]

    middle, depth = seen(centre)
    assert np.allclose(middle, 0.0, atol=1e-6) and np.isclose(depth, 10.0), "the centre, ten away"
    focal = 1.0 / np.tan(np.radians(FOV_DEGREES) / 2.0)
    assert np.allclose(seen(centre + [1.0, 0.0, 0.0])[0], [focal / 2.0 / 10.0, 0.0], atol=1e-6)
    assert np.allclose(seen(centre + [0.0, 1.0, 0.0])[0], [0.0, focal / 10.0], atol=1e-6)
    assert np.isclose(seen(centre + [0.0, 0.0, 4.0])[1], 6.0), "facing looks along z from the high side"
    assert matrix.dtype == np.float32 and matrix.shape == (4, 4)


def test_from_the_corner_the_lattices_index_0_corner_is_the_nearest():
    matrix = orbit_camera(np.zeros(3), 10.0, *CORNER, aspect=1.0)
    depth = {corner: float(matrix[3] @ np.array([*corner, 1.0]))
             for corner in [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]}
    assert min(depth, key=depth.get) == (-1, -1, -1)


def test_the_shown_kinds_are_three_different_choices():
    assert SHOWN == (SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH) == (1, 2, 3) and LEVELS == 65535
