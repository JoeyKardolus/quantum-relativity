"""In a line of cells the tracers cross a joint with the fluid, and each cell is drawn at its size."""
import numpy as np

from physics.engine import three_cells
from physics.view.logic.line_streaks import AT_MEAN, line_streaks, placed_line
from physics.view.logic.tracers import lay_tracers, step_carrying


def _cells(tracers) -> np.ndarray:
    """How many tracers lie in each of the three cells now."""
    return np.bincount((tracers.tail.numpy()[tracers.head][:, 1] // 6).astype(int), minlength=3)


def test_tracers_go_from_the_fuller_volume_to_the_other_and_the_joint_keeps_its_own():
    store = three_cells()
    tracers = lay_tracers(store, 0, 3000, cells=3)
    start = _cells(tracers)
    assert abs(start[0] - 1500) < 100 and abs(start[2] - 500) < 100  # laid by quanta: 3, 2, 1
    for done in range(300):
        step_carrying(store, tracers, 0, mark=done % 4 == 0, cells=3)
    after, place = _cells(tracers), tracers.tail.numpy()[tracers.head]
    assert after[1] == start[1] and after.sum() == 3000
    assert after[2] > start[2] + 300 and abs(after[0] - after[2]) < 300  # as the quanta: evened
    assert place.min() >= 0 and place[:, 1].max() < 18 and place[:, [0, 2]].max() <= 6


def test_each_cell_is_drawn_at_its_size_the_joint_s_way_below_0():
    store = three_cells()
    placed = placed_line(store, 3).numpy()
    assert np.allclose(placed[:, 1], [2 / 3, 1, 2]) and placed[:, 2].tolist() == [1, -1, 1]
    assert np.allclose(placed[:, 0], [-1.5, -2 / 3, 5 / 6])  # touching, the line's middle at 0
    tracers = lay_tracers(store, 0, 600, cells=3)
    for done in range(8):
        step_carrying(store, tracers, 0, mark=True, cells=3)
    drawn = line_streaks(tracers, 6, placed_line(store, 3)).numpy()[-1]
    cell = (tracers.tail.numpy()[tracers.head][:, 1] // 6).astype(int)
    across_picture = drawn[:, 0] / 6 - 0.5
    assert across_picture[cell == 0].max() < across_picture[cell == 1].min() + 0.02
    assert across_picture[cell == 1].max() < across_picture[cell == 2].min() + 0.02
    assert drawn[cell == 1, 3].max() <= 0 < drawn[cell == 1, 3].min() + 1 and drawn[cell != 1, 3].min() >= 0
    for one in range(3):  # one rule for both colours: fluid at its cell's mean speed is AT_MEAN
        ways = line_streaks(tracers, 6, placed_line(store, 3)).numpy()[1:, cell == one, 3]
        assert abs(np.sqrt((ways ** 2).mean()) - AT_MEAN) < 0.1 * AT_MEAN
