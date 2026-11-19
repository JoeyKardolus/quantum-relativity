"""A growing cluster is drawn evenly: new cells get tracers, the flat cut shows the middle layer only."""
import numpy as np

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.view.logic.scene import Seen, advance, cut, drawn, trace


def _slots(seen: Seen) -> np.ndarray:
    """The slot each tracer lies in now."""
    return (seen.tracers.tail.numpy()[seen.tracers.head][:, 1] // 6).astype(int)


def test_new_cells_get_their_share_of_the_tracers_and_the_flat_cut_shows_the_middle_layer():
    store = first_volume(layers=3)
    seen = Seen(trace(store))
    while store.count < 60:
        step(store)
    marks = drawn(store, seen).numpy()[-1]
    slots, count = _slots(seen), store.count
    held = np.bincount(slots, minlength=count)
    assert not cut(store, seen) and held[1:].min() > 0.3 * 32000 / count  # every cell is seen
    assert np.isfinite(marks).all() and (np.abs(marks[:, :3] / 6 - 0.5) < 2.5).all()  # in the window
    assert (marks[held.cumsum()[0]:, 3] != 0).any() and seen.span > 1  # drawn smaller, to fit
    seen.whole = False  # the key C: the flat cut
    marks = drawn(store, seen).numpy()[-1]
    middle = steps_of(store.own_cell.numpy()[:count], store.side)[:, 2] == store.side // 2
    assert cut(store, seen) and 0 < middle.sum() < count and middle[_slots(seen)].all()
    assert (marks[:, 3] != 0).mean() > 0.6  # drawn; of the faintest by the axis only their share


def test_tracers_the_fluid_carries_off_the_flat_cut_are_laid_in_it_again_before_drawing():
    store = first_volume(layers=3)
    seen = Seen(trace(store), whole=False)
    while store.count < 60:
        step(store)
    drawn(store, seen)  # fresh tracers, in the cut only
    for _ in range(40):
        advance(store, seen)
    cells = store.own_cell.numpy()[:store.count]
    in_cut = lambda: (steps_of(cells, store.side)[:, 2] == store.side // 2)[_slots(seen)]  # noqa: E731
    assert 0 < (~in_cut()).sum()  # some crossed a joint out of the layer (measured: 8,896 of 32,000)
    marks = drawn(store, seen).numpy()
    assert in_cut().all() and (marks[:, :, 3] != 0).mean() > 0.6  # none is drawn off the cut, tail and all


def test_after_a_collapse_the_joint_is_drawn_as_plus_between_its_two_and_the_title_counts_it():
    from physics.engine.axiom_6_limits.logic.tests.test_collapse import _staged
    from physics.view.logic.placed import placed, settle
    from physics.view.logic.scene import title
    store, _before, joint, ends = _staged()
    seen = Seen(trace(store))
    assert np.isfinite(drawn(store, seen).numpy()[-1]).all() and "collapses 1" in title(store, seen)
    middle, size, joints, _volumes, _apart = settle(store, seen.middle)
    shapes = placed(store, middle, size, joints, 1.0).numpy()
    assert shapes[joint, 2, 2] == 1 and (shapes[:, 2, 2] == 1).sum() == 29  # the 28 volumes and it
    assert np.allclose(shapes[joint, 0], (shapes[ends[0], 0] + shapes[ends[1], 0]) / 2)  # red, between its two
