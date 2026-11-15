"""Cell edges and corners move: a starved cell takes the boxes along its start edges and corners.

In an even checkerboard one inner energy volume keeps a quarter of its energy (the rest
moved, exactly, into a neighbouring volume), so it wants 64 times its boxes. Its rim
boxes are those one box outside its start cube along one axis and on the cube's border
along another: such a box belongs to a joint and touches, face to face, the energy
volume diagonal to the starved one across that edge, so no single box move can give it
to the starved volume (A would touch A): one-box moves froze every inner edge and
corner. Plate moves shift an edge or corner whole, so after the start's size correction
the cell holds rim boxes, some of them at a corner (on the border along both other
axes); ABAB holds, every cell is one piece, and not one quantum is lost.
"""
import numpy as np

from physics.potts import Settings, size_correction, stats
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import cell_id, connected_everywhere, even_universe


@needs_gpu
def test_a_starved_cell_reaches_past_its_start_edges_and_corners():
    universe = even_universe(6, 4)
    starved = cell_id(6, 2, 2, 2)
    fed = cell_id(6, 2, 2, 4)
    owner = universe.owner.numpy()
    energy = universe.energy.numpy()
    kept = energy[owner == starved] // 4
    moved = int((energy[owner == starved] - kept).sum())
    energy[owner == starved] = kept
    spread, rest = divmod(moved, int(np.sum(owner == fed)))
    energy[owner == fed] += spread
    energy[tuple(np.argwhere(owner == fed)[0])] += rest
    universe.energy.assign(energy)
    start_box = np.argwhere(owner == starved)
    low, high = start_box.min(axis=0), start_box.max(axis=0)
    settings = Settings.all_on()
    size_correction(universe, settings)
    owner = universe.owner.numpy()
    places = np.argwhere(owner == starved)
    beyond = (places < low) | (places > high)
    on_border = (places == low) | (places == high)
    rim = (beyond.sum(axis=1) == 1) & (np.abs(np.clip(places, low, high) - places).max(axis=1) == 1)
    rim_edges = int(np.sum(rim & (on_border.sum(axis=1) >= 1)))
    rim_corners = int(np.sum(rim & (on_border.sum(axis=1) == 2)))
    end = stats(universe, settings)
    print(f"starved cell: {len(start_box)} -> {len(places)} boxes, {rim_edges} rim boxes along its edges, "
          f"{rim_corners} of them at a corner")
    assert rim_edges > 0
    assert rim_corners > 0
    assert end["abab_breaks"] == 0
    assert end["energy_drift"] == 0
    assert connected_everywhere(owner, np.flatnonzero(universe.alive.numpy() == 1))
