"""ABAB holds and neighbours never change: 200 steps of the uneven start with every rule on.

No two energy volumes ever touch, nor two joints; the set of touching cell pairs is the
start's (no joint collapsed and no volume tore in this run, which the test checks); every cell stays one
connected piece; and the GPU's contact table equals one recounted from the boxes.
"""
import numpy as np

from physics.potts import Settings, contact_pairs, contact_table, new_universe, stats, step
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import connected_everywhere


@needs_gpu
def test_no_like_cells_touch_and_the_neighbours_stay_the_same_for_200_steps():
    universe = new_universe(6, 4, seed=1, unevenness=0.3)
    settings = Settings.all_on()
    low, high, _ = contact_pairs(universe.owner.numpy())
    start_pairs = set(zip(low.tolist(), high.tolist()))
    for count in range(200):
        step(universe, settings)
        if count % 50 == 49:
            assert stats(universe, settings)["abab_breaks"] == 0
    end = stats(universe, settings)
    owner = universe.owner.numpy()
    low, high, _ = contact_pairs(owner)
    assert end["abab_breaks"] == 0
    assert end["collapses"] == 0
    assert end["splits"] == 0
    assert set(zip(low.tolist(), high.tolist())) == start_pairs
    assert connected_everywhere(owner, np.flatnonzero(universe.alive.numpy() == 1))
    neighbour, faces, slots = contact_table(owner, universe.cell_count, universe.slots)
    assert slots == universe.slots
    assert _entries(neighbour, faces, slots) == _entries(universe.neighbour.numpy(), universe.faces.numpy(), slots)


def _entries(neighbour: np.ndarray, faces: np.ndarray, slots: int) -> set:
    return {(slot // slots, int(neighbour[slot]), int(faces[slot])) for slot in np.flatnonzero(neighbour >= 0)}
