"""A growing cluster is drawn evenly: new cells get tracers, the flat cut shows the middle layer only."""
import numpy as np

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.view.logic.scene import Seen, cut, drawn, trace


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
    assert (marks[:, 3] != 0).all()
