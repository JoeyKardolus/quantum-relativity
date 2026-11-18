"""A grid's layer is round reds in blue, the same along each direction; the slab shows the joints between layers."""
import numpy as np
import pytest

from physics.engine.axiom_3_structure.logic.lay_grid import calm_grid
from physics.view.definitions.camera import ALONG, FOCAL
from physics.view.logic.placed import placed
from physics.view.logic.scene import drawn, eye_distance, trace
from physics.view.logic.tracers import MARKS, step_carrying

pyglet = pytest.importorskip("pyglet")
SIDE = 400


@pytest.fixture(scope="module")
def seen():
    """Draws a 2 x 2 x 2 grid after 24 steps: `seen(slab, yaw)` is the picture, `seen.apart` the pitch."""
    try:
        window = pyglet.window.Window(visible=False)
    except Exception as why:  # no screen to draw on here
        pytest.skip(f"no window: {why}")
    from physics.view.logic.picture import picture
    from physics.view.logic.swirl_painter import SwirlPainter

    def look(slab: int, yaw: float, pitch: float = 0.0, along: int = 2) -> np.ndarray:
        store = calm_grid(reds=(2, 2, 2))
        tracers = trace(store, 0, slab, along)
        for _ in range(24):
            step_carrying(store, tracers, 0, mark=True, cells=store.count)
        painter = SwirlPainter(MARKS, tracers.tail.shape[1])
        painter.thick = 2.0
        painter.load(drawn(store, tracers, 0, slab, along))
        look.apart = float(placed(store, 0).numpy()[:, 1, 0].max())
        return picture(painter, 6, yaw, pitch, eye_distance(store), SIDE, SIDE).astype(int)
    yield look
    window.close()


def _at(picture: np.ndarray, right: float, up: float, distance: float = 5.2) -> np.ndarray:
    """The mean colour of the few picture dots round a place in the flat cut, in cell edges."""
    column = int((1 + FOCAL * right / distance) * SIDE / 2)
    row = int((1 - FOCAL * up / distance) * SIDE / 2)
    return picture[row - 2:row + 3, column - 2:column + 3].mean(axis=(0, 1))


def test_along_each_direction_the_reds_are_round_with_a_dark_middle_and_blue_between(seen):
    for along in range(3):
        flat, half = seen(0, *ALONG[along], along), seen.apart / 2
        for right, up in ((half, half), (-half, half), (half, -half), (-half, -half)):  # each red
            ring = np.array([_at(flat, right + 0.3 * np.cos(angle), up + 0.3 * np.sin(angle))
                             for angle in np.arange(8) * np.pi / 4])
            assert ring[:, 0].min() > 90 and ring[:, 2].max() < 60, along
            assert ring[:, 0].max() < 1.4 * ring[:, 0].min(), along      # round
            assert _at(flat, right, up)[0] < 0.6 * ring[:, 0].min(), along  # its middle dark
        for right, up in ((0.0, 0.0), (0.15, 0.15), (0.2, half), (half, -0.2)):
            red, _green, blue = _at(flat, right, up)
            assert blue > 80 and blue > 2 * red, (along, right, up)      # blue between the reds


def test_the_slab_shows_the_joints_from_layer_to_layer(seen):
    from_the_side = seen(2, np.pi / 2)  # the two layers stand left and right; only joints between
    between = from_the_side[80:320, SIDE // 2 - 3:SIDE // 2 + 4].mean(axis=(0, 1))
    assert between[2] > 20 and between[2] > 2 * between[0]  # the web: joints cut both ways
    layer = int(FOCAL * seen.apart / 2 / 5.2 * SIDE / 2)  # picture dots from the middle to a layer
    assert from_the_side[80:320, SIDE // 2 - layer - 3:SIDE // 2 - layer + 4, 0].mean() > 2 * between[0]
