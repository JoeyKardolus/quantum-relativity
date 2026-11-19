"""The bigger a volume, the bigger the share it leaks; it leaks alike toward all 26 directions."""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_4_crossing.definitions.leak_share import leak_share, toward


@wp.kernel
def _shares(calms: wp.array(dtype=wp.float64), found: wp.array(dtype=wp.float64)):
    found[wp.tid()] = leak_share(calms[wp.tid()] * wp.float64(648000.0), wp.float64(648000.0))


def test_the_share_grows_with_the_size_and_never_passes_the_whole():
    found = wp.zeros(5, dtype=wp.float64, device="cpu")
    wp.launch(_shares, dim=5, device="cpu",
              inputs=[wp.array([100, 2, 1, 0.5, 0.01], dtype=wp.float64, device="cpu"), found])
    shares = found.numpy()
    assert shares[2] == 0.5 and np.isclose(shares[1], 1 / 3)  # a calm volume: half a turn
    assert (np.diff(shares) > 0).all() and shares[-1] < 1  # the bigger, the more; never the whole
    assert np.isclose(shares[0], 0.01, rtol=0.01)  # packed: in step with its size, 1/100


@wp.kernel
def _picks(cell: wp.int32, side: wp.int32, found: wp.array(dtype=wp.int32)):
    found[wp.tid()] = toward(cell, side, (wp.float64(wp.tid()) + wp.float64(0.5)) / wp.float64(2600.0))


def test_the_draw_picks_each_of_the_26_places_round_a_volume_alike():
    side, middle, corner = 5, 62, 0
    found = wp.zeros(2600, dtype=wp.int32, device="cpu")
    wp.launch(_picks, dim=2600, inputs=[middle, side, found], device="cpu")
    places, times = np.unique(found.numpy(), return_counts=True)
    assert len(places) == 26 and (times == 100).all() and middle not in places
    assert (np.abs(steps_of(places, side) - 2).max(axis=1) == 2).all()  # two steps on: volumes' places
    wp.launch(_picks, dim=2600, inputs=[corner, side, found], device="cpu")
    assert (found.numpy() == -1).sum() == 1900  # 19 of 26 lie past the room's end
