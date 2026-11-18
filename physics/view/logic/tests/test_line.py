"""Tracers cross a joint with the fluid; volumes are drawn round and joints fill between them."""
import numpy as np

from physics.engine import three_cells
from physics.engine.axiom_3_structure.logic.lay_grid import calm_grid
from physics.view.logic.line_streaks import AT_MEAN, line_streaks
from physics.view.logic.placed import placed
from physics.view.logic.scene import trace
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


def test_round_volumes_are_drawn_with_the_joints_filling_the_room_between_them():
    store = calm_grid(reds=(2, 2, 2))
    tracers = trace(store, 0)  # the flat cut through the first layer: 4 volumes, 8 joints
    for done in range(24):
        step_carrying(store, tracers, 0, mark=True, cells=store.count)
    shapes, streaks = placed(store, 0).numpy(), line_streaks(tracers, 6, placed(store, 0)).numpy()
    spot = streaks[-1, :, :3] / 6 - 0.5  # where each tracer is drawn: right, up, toward the eye
    slot = (tracers.tail.numpy()[tracers.head][:, 1] // 6).astype(int)
    cells = store.own_cell.numpy()[:store.count]
    plus, deep = store.cell_quanta.numpy()[cells] > 0, np.unravel_index(cells, store.grid)[2]
    volumes, joints = np.flatnonzero(plus & (deep == 0)), np.flatnonzero(~plus & (deep == 0))
    assert set(slot) == set(volumes) | set(joints) and len(joints) == 8  # the cut's cells, and
    assert np.abs(spot[:, 2]).max() < 0.3 and len(volumes) == 4  # flat: the middle layers of boxes
    for one in volumes:  # round: inside half its size from its middle
        assert np.linalg.norm(spot[slot == one] - shapes[one, 0], axis=1).max() <= 0.5 + 1e-4
        assert streaks[1:, slot == one, 3].min() >= 0
    for one in joints:  # blue
        assert streaks[1:, slot == one, 3].max() <= 0
        for other in volumes:  # it fills what the round volumes leave: never inside one
            assert np.linalg.norm(spot[slot == one] - shapes[other, 0], axis=1).min() >= 0.5 - 1e-3
    thin = shapes[joints[0], 1, 0] - 1  # the volumes' distance less two radii; fresh: 0.52
    right = joints[(shapes[joints, 2, 1] == 1) & (shapes[joints, 2, 0] == 0)][0]  # no wrap
    own = spot[slot == right] - shapes[right, 0]
    assert abs(np.ptp(own[np.abs(own[:, 1]) < 0.03, 0]) - thin) < 0.08 and 0.4 < thin < 0.55
    assert np.ptp(own[(np.abs(own[:, 1]) > 0.3) & (np.abs(own[:, 1]) < 0.4), 0]) > 1.3 * thin
    near = np.linalg.norm(spot[:, :2], axis=1) < 0.15  # the middle: where four volumes meet
    assert len(set(slot[near]) & set(joints)) == 4 and not set(slot[near]) & set(volumes)
    sides = spot[slot == joints[(shapes[joints, 2, 1] == 1) & (shapes[joints, 2, 0] != 0)][0], 0]
    assert (sides > 1.0).sum() > 100 and (sides < -1.0).sum() > 100 and (np.abs(sides) < 1).sum() == 0
    jump = np.linalg.norm(streaks[1:, :, :3] - streaks[:-1, :, :3], axis=2) > 6
    way_round = np.abs(streaks[1:, :, 3][jump]).max() + np.abs(streaks[:-1, :, 3][jump]).max()
    assert jump.any() and way_round == 0  # the way round the wrap is not drawn


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
