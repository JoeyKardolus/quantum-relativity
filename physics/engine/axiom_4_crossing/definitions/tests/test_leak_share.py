"""The bigger a volume, the bigger the share it leaks, never more than a seventh; it leaks alike
toward all 26 directions."""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_4_crossing.definitions.leak_share import leak_share, toward


@wp.kernel
def _shares(calms: wp.array(dtype=wp.float64), found: wp.array(dtype=wp.float64)):
    found[wp.tid()] = leak_share(calms[wp.tid()] * wp.float64(648000.0), wp.float64(648000.0))


def test_a_thinner_volume_leaks_a_bigger_share_and_none_more_than_a_seventh():
    found = wp.zeros(6, dtype=wp.float64, device="cpu")
    wp.launch(_shares, dim=6, device="cpu",
              inputs=[wp.array([1000, 2, 1, 0.5, 1 / 60, 1e-6], dtype=wp.float64, device="cpu"), found])
    shares = found.numpy()
    assert shares[2] == 1 / 8 and np.isclose(shares[1], 1 / 9)  # a calm volume: an eighth a turn
    assert (np.diff(shares) > 0).all() and shares[-1] < 1 / 7  # the bigger, the more; never a seventh
    assert np.isclose(shares[0], 0.001, rtol=0.01)  # packed: in step with its size, 1/1000
    assert shares[4] < 1 / 7  # the thinnest whirl keeps more than six sevenths of itself a turn


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
