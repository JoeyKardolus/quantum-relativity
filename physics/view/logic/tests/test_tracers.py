"""The tracers go round the swirl's axis with the fluid, and stand still when the fluid does."""
import numpy as np

from physics.engine import one_cell
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.engine.axiom_3_structure.logic.give_back import give_back
from physics.view.logic.tracers import lay_tracers, step_carrying


def _across_the_axis(tracers, across: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each tracer's angle round the cell's third axis, its distance from it, its place along it."""
    place = tracers.tail.numpy()[tracers.head] - across / 2
    return np.arctan2(place[:, 1], place[:, 0]), np.hypot(place[:, 0], place[:, 1]), place[:, 2]


def test_the_tracers_go_round_the_axis_and_stay_spread_as_the_fluid():
    store = one_cell()
    tracers = lay_tracers(store, 0, 400)
    angle, distance, along = _across_the_axis(tracers, store.across)
    turned = np.zeros(400)
    for done in range(30):
        step_carrying(store, tracers, 0, mark=done % 5 == 0)
        now = _across_the_axis(tracers, store.across)[0]
        turned += (now - angle + np.pi) % (2 * np.pi) - np.pi
        angle = now
    _angle, distance_after, along_after = _across_the_axis(tracers, store.across)
    assert (np.sign(turned) == np.sign(np.median(turned))).mean() > 0.95  # all the same way round
    assert abs(np.median(turned)) > 1.5                                 # a good part of a round
    assert abs(distance_after.mean() - distance.mean()) < 0.1           # not into the core, not out
    assert np.abs(along_after - along).mean() < 0.1                     # not along the axis
    assert tracers.head == 6 and not np.array_equal(*tracers.tail.numpy()[[5, 6]])  # tails kept


def test_fluid_that_stands_still_moves_no_tracer():
    store = one_cell()
    tracers = lay_tracers(store, 0, 100)
    store.box_quanta.fill_(3000)
    store.box_phase.fill_(0.5)
    before = tracers.tail.numpy()[tracers.head].copy()
    for _ in range(5):
        step_carrying(store, tracers, 0, mark=False)
    assert np.abs(tracers.tail.numpy()[tracers.head] - before).max() < 1e-4


def test_a_tracer_from_anywhere_is_carried_inside_the_block():
    store = one_cell()
    tracers = lay_tracers(store, 0, 6)
    wild = [[-1e12, 3, 3], [3, 1e12, 3], [3, 3, -5], [1e30, -1e30, 1e30], [6, 6, 6], [0, 0, 0]]
    tracers.tail.assign(np.tile(np.array(wild, dtype=np.float32), (tracers.tail.shape[0], 1, 1)))
    step_carrying(store, tracers, 0, mark=True)
    place = tracers.tail.numpy()[tracers.head]
    assert np.isfinite(place).all() and place.min() >= 0 and place.max() <= store.across


def test_a_calm_cell_carries_the_same_with_its_boxes_asked_once():
    store = one_cell()
    assert give_back(store, 0)
    asked_once, asked_each_step = lay_tracers(store, 0, 200), lay_tracers(store, 0, 200)
    boxes = boxes_of(store, 0)
    for _ in range(12):
        step_carrying(store, asked_once, 0, mark=True, boxes=boxes)
        store.step -= 1  # the same step again, for the other tracers
        step_carrying(store, asked_each_step, 0, mark=True)
    moved = np.abs(asked_once.tail.numpy()[12] - asked_once.tail.numpy()[0]).max()
    assert moved > 0.5 and np.allclose(asked_once.tail.numpy(), asked_each_step.tail.numpy(), atol=1e-3)
