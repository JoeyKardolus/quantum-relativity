"""Tracers cross a joint with the fluid; volumes are drawn round and joints fill between them."""
import numpy as np

from physics.engine import three_cells
from physics.engine.axiom_3_structure.logic.lay_square import four_volumes
from physics.view.logic.line_streaks import AT_MEAN, line_streaks
from physics.view.logic.placed import placed
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


def test_round_volumes_are_drawn_with_blue_pinched_between_them_and_wide_at_the_corners():
    store = four_volumes()
    tracers = lay_tracers(store, 0, 8000, cells=8)
    for done in range(8):
        step_carrying(store, tracers, 0, mark=True, cells=8)
    shapes = placed(store).numpy()
    streaks = line_streaks(tracers, 6, placed(store)).numpy()
    spot = streaks[-1, :, :2] / 6 - 0.5  # where each tracer is drawn: right, up
    slot = (tracers.tail.numpy()[tracers.head][:, 1] // 6).astype(int)
    volumes, joints = [0, 2, 5, 7], [1, 3, 4, 6]
    for one in volumes:  # round: inside half its size from its middle
        assert np.hypot(*(spot[slot == one] - shapes[one, 0, :2]).T).max() <= shapes[one, 1, 0] / 2 + 1e-4
        assert streaks[1:, slot == one, 3].min() >= 0
    for one in joints:
        own, axis = spot[slot == one], int(shapes[one, 2, 1])
        assert streaks[1:, slot == one, 3].max() <= 0  # blue
        for other in volumes:  # it fills what the round volumes leave: never inside one
            away = np.hypot(*(own - shapes[other, 0, :2]).T)
            assert away.min() >= shapes[other, 1, 0] / 2 - 1e-3
        along, side = own[:, 1 - axis] - shapes[one, 0, 1 - axis], np.abs(own[:, axis] - shapes[one, 0, axis])
        thin = shapes[one, 1, 0] - shapes[one, 1, 1] - shapes[one, 1, 2]
        assert abs(np.ptp(along[side < 0.03]) - thin) < 0.05 and 0.35 < thin < 0.5  # fresh: 0.40
        assert np.ptp(along[(side > 0.3) & (side < 0.5)]) > 1.3 * thin  # wider toward the corners
    middle = np.hypot(*spot[np.isin(slot, joints)].T)
    assert middle.min() < 0.05 and np.hypot(*spot[np.isin(slot, volumes)].T).min() > 0.3  # blue where four meet


def test_each_cell_is_drawn_at_its_place_the_joint_s_way_below_0():
    store = three_cells()
    shapes = placed(store).numpy()
    apart = (3 + np.pi / 12 * (1.5 ** -3 + 0.5 ** -3)) ** (1 / 3)
    assert np.allclose(shapes[[0, 2], 0, 0], [-apart / 2, apart / 2]) and np.allclose(shapes[[0, 2], 1, 0], [2 / 3, 2])
    assert np.allclose(shapes[1, 1], [apart, 1 / 3, 1]) and shapes[1, 0, 0] == 0  # midway
    tracers = lay_tracers(store, 0, 600, cells=3)
    for done in range(8):
        step_carrying(store, tracers, 0, mark=True, cells=3)
    drawn = line_streaks(tracers, 6, placed(store)).numpy()[-1]
    cell = (tracers.tail.numpy()[tracers.head][:, 1] // 6).astype(int)
    across_picture = drawn[:, 0] / 6 - 0.5
    assert across_picture[cell == 0].max() < across_picture[cell == 2].min()  # the joint lies between
    assert drawn[cell == 1, 3].max() <= 0 < drawn[cell == 1, 3].min() + 1 and drawn[cell != 1, 3].min() >= 0
    for one in range(3):  # one rule for both colours: fluid at its cell's mean speed is AT_MEAN
        ways = line_streaks(tracers, 6, placed(store)).numpy()[1:, cell == one, 3]
        assert abs(np.sqrt((ways ** 2).mean()) - AT_MEAN) < 0.1 * AT_MEAN
