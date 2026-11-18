"""Pictures as a whole: the whole grid drawn by the GPU off screen, on blocks whose picture is known.

Needs a display and OpenGL 3.3 (a hidden window). One cube is drawn alone; the pixel in
the middle of the picture must be the cell's hue times the shade of the face it shows.
"""
import numpy as np
import pytest

from physics.view.colours.definitions.palette import JOINT_BLUE, SOLID, VOLUME_RED, WHOLE_POWER
from physics.view.drawing.definitions.block_faces import FACE_SHADE
from physics.view.layout.definitions.pieces import Pieces
from physics.view.tests.gpu_cases import need_display

LOW, HIGH = np.full(3, -1.0, dtype=np.float32), np.full(3, 1.0, dtype=np.float32)


def _cubes(centres, half, kinds, packings) -> Pieces:
    unit = np.array([[c & 1, c >> 1 & 1, c >> 2 & 1] for c in range(8)], dtype=np.float32) * 2.0 - 1.0
    corners = np.stack([np.asarray(centre, dtype=np.float32) + half * unit for centre in centres])
    count = len(centres)
    return Pieces(corners, np.arange(count), np.array(kinds, dtype=np.int8), np.array(packings, dtype=np.float32),
                  np.zeros((count, 3), dtype=np.int64), 4, LOW, HIGH)


@pytest.fixture(scope="module")
def draw():
    need_display()
    try:
        from physics.view.pictures.logic.whole_picture import WholePicture
        return WholePicture(220, 160)
    except Exception as raised:
        pytest.skip(f"no OpenGL window here: {raised}")


def _shades(hue) -> list:
    return [np.round(np.array(hue) * shade) for shade in FACE_SHADE]


def test_a_solid_cell_is_its_hue_shaded_by_the_face_it_shows_on_black(draw):
    picture = draw(_cubes([(0, 0, 0)], 0.5, [1], [SOLID]), LOW, HIGH).astype(int)
    assert picture.shape == (160, 220, 3)
    middle = picture[80, 110]
    assert any(np.abs(middle - shade).max() <= 2 for shade in _shades(VOLUME_RED)), f"middle pixel {middle}"
    assert picture[0, 0].sum() == 0 and picture[-1, -1].sum() == 0, "black round the grid"
    seen = {tuple(pixel) for pixel in picture.reshape(-1, 3) if pixel.sum()}
    assert 2 <= len({pixel[0] for pixel in seen}) and all(
        any(np.abs(np.array(pixel) - shade).max() <= 2 for shade in _shades(VOLUME_RED)) for pixel in seen), \
        "nothing but the three shades of red: hard edges, no light of its own"


def test_an_emptied_cell_is_not_there_and_the_starts_mean_is_haze(draw):
    assert draw(_cubes([(0, 0, 0)], 0.5, [1], [0.0]), LOW, HIGH).sum() == 0
    haze = draw(_cubes([(0, 0, 0)], 0.5, [-1], [1.0]), LOW, HIGH).astype(int)[80, 110]
    through_two_faces = 1.0 - (1.0 - 0.25 ** WHOLE_POWER) ** 2
    assert 0 < haze[2] <= np.ceil(JOINT_BLUE[2] * through_two_faces) + 1, f"middle pixel {haze}"


def test_the_near_solid_cell_hides_the_far_one_whatever_order_they_come_in(draw):
    from physics.view.drawing.definitions.camera_marks import CORNER
    toward_eye = np.array([np.cos(CORNER[1]) * np.sin(CORNER[0]), np.sin(CORNER[1]), np.cos(CORNER[1]) * np.cos(CORNER[0])])
    near, far = 0.45 * toward_eye, -0.45 * toward_eye
    for centres, kinds in (((near, far), (1, -1)), ((far, near), (-1, 1))):
        middle = draw(_cubes(centres, 0.3, kinds, [SOLID, SOLID]), LOW, HIGH).astype(int)[80, 110]
        assert middle[0] > 150 and middle[2] < 40, f"the red one is nearer: middle pixel {middle}"
