"""Cell edges and corners move: a starved energy volume grows past its start edges and corners, still wrapped.

In an even start one inner energy volume keeps a quarter of its energy (the rest moved
into another volume), so it wants 64 times its boxes. Plate moves shift a border, an
edge or a corner whole (one-box moves froze every inner edge and corner on the start
lattice), so after the start's size correction the volume holds boxes beyond its start
cube along two axes (taken from the edge joints round it) and along all three (from
the corner joints); no energy volume touches another or empty space (26 neighbours),
every cell is one piece, and the total energy holds to rounding.
"""
import numpy as np

from physics.potts import Settings, size_correction, stats
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import cell_id, connected_everywhere, even_universe, move_energy, wrapped_everywhere


@needs_gpu
def test_a_starved_volume_reaches_past_its_start_edges_and_corners():
    universe = even_universe(7, 4)
    starved = cell_id(7, 3, 3, 3)
    fed = cell_id(7, 1, 1, 1)
    start_box = np.argwhere(universe.owner.numpy() == starved)
    low, high = start_box.min(axis=0), start_box.max(axis=0)
    move_energy(universe, starved, fed, 0.25)
    settings = Settings.all_on()
    start_total = stats(universe, settings)["total_energy"]
    size_correction(universe, settings)
    owner = universe.owner.numpy()
    places = np.argwhere(owner == starved)
    beyond = ((places < low) | (places > high)).sum(axis=1)
    end = stats(universe, settings)
    print(f"starved volume: {len(start_box)} -> {len(places)} boxes, {int(np.sum(beyond == 2))} past an edge, "
          f"{int(np.sum(beyond == 3))} past a corner")
    assert np.sum(beyond == 2) > 0
    assert np.sum(beyond == 3) > 0
    assert end["wrap_breaks"] == 0
    assert wrapped_everywhere(owner, universe.kind.numpy())
    assert abs(end["total_energy"] - start_total) < 1e-10 * start_total
    assert connected_everywhere(owner, np.flatnonzero(universe.alive.numpy() == 1))
