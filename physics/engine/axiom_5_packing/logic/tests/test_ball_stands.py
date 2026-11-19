"""The ball the engine grew (`ball.py`): it stands, its first volume whole in a thin halo, and that volume's pull rules it."""
import numpy as np

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_5_packing.logic.ball import PAIR, staged
from physics.engine.axiom_5_packing.logic.between import between, held_of, volumes_of
from physics.engine.axiom_5_packing.logic.weighed import volumes_held


def test_the_ball_stands_its_first_volume_whole_in_a_thin_halo(grown_patch):
    held = volumes_held(grown_patch) * (held_of(grown_patch)[::2, ::2, ::2] > 0)  # standing volumes only
    out = np.abs(np.indices(held.shape) - held.shape[0] // 2).max(axis=0)
    layer = [held[(out == one) & (held > 0)] for one in range(4)]
    print(f"the ball at {grown_patch.step / 25:.0f} turns: " + "; ".join(
        f"layer {one}: {len(has)} volumes, {has.sum():.2f} calm volumes, median {np.median(has):.3f}" for one, has in enumerate(layer)))
    assert layer[0][0] > 1630 and layer[0][0] > 0.99 * held.sum()  # the first volume: 1,640 of 1,657, the rest is the leak's
    assert len(layer[1]) == 26 and len(layer[2]) > 60 and len(layer[3]) >= 1  # grown layer by layer
    assert np.median(layer[1]) > 2 * np.median(layer[2]) and np.median(layer[2]) < 0.1  # thinner outward; the rim is thin, not calm


def test_the_first_volume_s_pull_on_a_region_is_thousands_of_times_the_other_region_s():
    store = staged(PAIR)
    held = held_of(store)
    cells = volumes_of(store, held)
    at = steps_of(cells, store.side) // 2 - store.side // 4
    apart = np.linalg.norm(between(held, steps_of(cells, store.side)), axis=2)
    here, first = np.flatnonzero((at == PAIR[1]).all(axis=1))[0], np.flatnonzero((at == 0).all(axis=1))[0]
    other = np.flatnonzero(np.abs(at - PAIR[0]).sum(axis=1) <= 1)
    energy = held[tuple(steps_of(cells, store.side).T)]
    by_first, by_other = energy[first] / apart[here, first] ** 2, (energy[other] / apart[here, other] ** 2).sum()
    print(f"pull on the region's middle: {by_first:.3f} by the first volume, {by_other:.2e} by the other region, "
          f"{apart[here, first]:.0f} and {apart[here, other].mean():.0f} calm lengths off")
    assert by_first > 1000 * by_other > 0  # 0.33 against 3e-05: the ball is one packed volume in a thin halo
