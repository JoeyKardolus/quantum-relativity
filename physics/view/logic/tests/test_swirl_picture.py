"""The picture is red where the fluid turns and shows no box, and any view at all can be drawn."""
import numpy as np
import pytest

from physics.engine import one_cell
from physics.view.definitions.camera import FARTHEST, NEAREST
from physics.view.logic.streaks import streaks
from physics.view.logic.tracers import MARK_EVERY, MARKS, lay_tracers, step_carrying

pyglet = pytest.importorskip("pyglet")


@pytest.fixture(scope="module")
def drawn():
    """A painter loaded with a swirl that has turned for 160 steps, and its cell's boxes across."""
    try:
        window = pyglet.window.Window(visible=False)
    except Exception as why:  # no screen to draw on here
        pytest.skip(f"no window: {why}")
    from physics.view.logic.swirl_painter import SwirlPainter
    store, painter = one_cell(), SwirlPainter(MARKS, 400)
    tracers = lay_tracers(store, 0, 400)
    for done in range(160):
        step_carrying(store, tracers, 0, mark=done % MARK_EVERY == 0)
    painter.load(streaks(tracers))
    yield painter, store.across
    window.close()


def _down_the_axis(drawn, distance: float = 2.6, width: int = 400, height: int = 400):
    from physics.view.logic.picture import picture
    return picture(drawn[0], drawn[1], 0.0, 0.0, distance, width, height).astype(int)


def test_the_red_is_the_swirl_and_no_box_shows(drawn):
    seen = _down_the_axis(drawn)
    red = seen[..., 0]
    assert seen[:60].max() == 0 and seen[:, :60].max() == 0            # black round the cell
    assert (red >= seen[..., 1]).all() and seen[..., 1].max() < 60     # red and nothing else
    corners = np.concatenate([red[80:104, 80:104], red[296:320, 80:104],
                              red[80:104, 296:320], red[296:320, 296:320]])
    assert red[150:250, 150:250].mean() > 60                           # the turning fluid is red
    assert corners.mean() < 0.1 * red[150:250, 150:250].mean()         # its still corners: no box


def test_any_view_draws_and_harms_nothing(drawn):
    first = _down_the_axis(drawn)
    for distance in (0.0, NEAREST, 10.0, FARTHEST, 1.0e15, 1.0e30):
        for width, height in ((1, 1), (333, 7), (64, 48)):
            assert _down_the_axis(drawn, distance, width, height).shape == (height, width, 3)
    assert np.array_equal(_down_the_axis(drawn), first)
