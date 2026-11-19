"""One ring alone shows its empty core: a ring closed inside the ball, the jet up the middle."""
import numpy as np

from physics.engine import one_cell
from physics.view.logic.lay_tracers import lay_tracers
from physics.view.logic.scene import Seen, drawn as marks


def test_one_ring_alone_shows_its_empty_core_a_ring_closed_inside_the_ball():
    from physics.view.definitions.ring_place import AT
    store = one_cell()
    spots = marks(store, Seen(lay_tracers(store, 0, 20000), ring=2)).numpy()[-1]
    place = spots[spots[:, 3] != 0, :3] / store.across - 0.5  # in a ball 1 wide: ring 2 about the third
    off_core = np.hypot(np.hypot(place[:, 0], place[:, 1]) - AT, place[:, 2])
    assert (off_core < 0.04).sum() == 0 and (off_core < 0.08).sum() > 200  # empty, fluid round it
    assert np.linalg.norm(place, axis=1).max() < 0.5 + 1e-4             # and all of it inside the ball
    assert np.hypot(place[:, 0], place[:, 1]).min() < 0.02              # the jet runs up the middle
