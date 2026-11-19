"""Tracers cross a joint with the fluid; each cell is drawn where its joints put it."""
import numpy as np

from physics.engine import three_cells
from physics.view.logic.line_streaks import AMOUNT, line_streaks
from physics.view.logic.placed import placed, settle
from physics.view.logic.lay_tracers import lay_tracers
from physics.view.logic.tracers import step_carrying


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


def _placed(store):
    """The cells' shapes at their true sizes, settled from nothing."""
    return placed(store, *settle(store, np.zeros((0, 3)))[:3], 1.0)


def test_each_cell_is_drawn_at_its_place_the_joint_s_way_below_0():
    store = three_cells()
    shapes = _placed(store).numpy()
    apart = 0.5 / 1.5 + 1 + 0.5 / 0.5  # half a width, the joint's thickness, half the other width
    assert np.allclose(shapes[[0, 2], 0, 0], [-apart / 2, apart / 2]) and np.allclose(shapes[[0, 2], 1, 0], [2 / 3, 2])
    assert np.allclose(shapes[1, 1], [apart, 1 / 3, 1]) and shapes[1, 0, 0] == 0  # midway
    tracers = lay_tracers(store, 0, 600, cells=3)
    for done in range(8):
        step_carrying(store, tracers, 0, mark=True, cells=3)
    drawn = line_streaks(tracers, 6, _placed(store)).numpy()[-1]
    cell = (tracers.tail.numpy()[tracers.head][:, 1] // 6).astype(int)
    across_picture = drawn[:, 0] / 6 - 0.5
    assert across_picture[cell == 0].max() < across_picture[cell == 2].min()  # the joint lies between
    assert drawn[cell == 1, 3].max() <= 0 < drawn[cell == 1, 3].min() + 1 and drawn[cell != 1, 3].min() >= 0
    assert np.allclose(drawn[cell == 1, 3], -AMOUNT)  # one rule: every mark of a joint counts alike,
    inside = tracers.tail.numpy()[tracers.head][cell == 0] / 6 - 0.5  # and a ball's by its even share
    even = (np.abs(inside).max(axis=1) / np.linalg.norm(inside, axis=1)) ** 3 / (np.pi / 6)
    assert np.allclose(drawn[cell == 0, 3], AMOUNT * even, rtol=1e-3) and np.ptp(even) > 1
