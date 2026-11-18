"""A grid's flat cut is round reds in blue with no black between; the slab shows the joints between layers."""
import numpy as np
import pytest

from physics.engine.axiom_3_structure.logic.lay_grid import calm_grid
from physics.view.definitions.camera import FOCAL
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

    def look(slab: int, yaw: float) -> np.ndarray:
        store = calm_grid(reds=(2, 2, 2))
        tracers = trace(store, 0, slab)
        for _ in range(24):
            step_carrying(store, tracers, 0, mark=True, cells=store.count)
        painter = SwirlPainter(MARKS, tracers.tail.shape[1])
        painter.thick, painter.dots = 2.0, True
        painter.load(drawn(store, tracers, 0, slab))
        look.apart = float(placed(store, 0).numpy()[:, 1, 0].max())
        return picture(painter, 6, yaw, 0.0, eye_distance(store), SIDE, SIDE).astype(int)
    yield look
    window.close()


def _at(picture: np.ndarray, right: float, up: float, distance: float = 5.2) -> np.ndarray:
    """The mean colour of the few picture dots round a place in the flat cut, in cell edges."""
    column = int((1 + FOCAL * right / distance) * SIDE / 2)
    row = int((1 - FOCAL * up / distance) * SIDE / 2)
    return picture[row - 2:row + 3, column - 2:column + 3].mean(axis=(0, 1))


def test_the_flat_cut_is_round_reds_in_blue_with_no_black_inside(seen):
    cut, half = seen(0, 0.0), seen.apart / 2
    blue_places = [(0.0, 0.0), (0.15, 0.15), (-0.2, 0.1),             # where four reds meet
                   (0.56 - half, half), (half, half - 0.56),          # just off a red's rim
                   (0.2, half), (half, -0.2)]                         # midway to its neighbour
    for right, up in blue_places:
        red, _green, blue = _at(cut, right, up)
        assert blue > 80 and blue > 2 * red, (right, up)
    ring = [_at(cut, half + 0.3 * np.cos(angle), half + 0.3 * np.sin(angle))
            for angle in np.arange(8) * np.pi / 4]  # a red's middle ring, along and between the edges
    assert min(colour[0] for colour in ring) > 120 and max(colour[2] for colour in ring) < 60
    assert max(colour[0] for colour in ring) < 1.3 * min(colour[0] for colour in ring)  # round
    inside = cut[int(SIDE / 2 - 100):int(SIDE / 2 + 100), int(SIDE / 2 - 100):int(SIDE / 2 + 100)]
    assert (inside.max(axis=2) < 25).mean() < 0.03  # no black inside but the swirls' own cores


def test_the_slab_shows_the_joints_from_layer_to_layer(seen):
    from_the_side = seen(2, np.pi / 2)  # the two layers stand left and right; only joints between
    between = from_the_side[80:320, SIDE // 2 - 3:SIDE // 2 + 4].mean(axis=(0, 1))
    assert between[2] > 40 and between[2] > 2 * between[0]
    layer = int(FOCAL * seen.apart / 2 / 5.2 * SIDE / 2)  # picture dots from the middle to a layer
    assert from_the_side[80:320, SIDE // 2 - layer - 3:SIDE // 2 - layer + 4, 0].mean() > 2 * between[0]
