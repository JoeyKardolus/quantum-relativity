"""Volume split: an energy volume past the split size tears in two around a new joint (rule 8).

In an even checkerboard of 6**3 cells of 6**3 boxes one inner energy volume keeps a twentieth of its energy (the
rest moved, exactly, into another volume), so it wants thousands of boxes; the start's
size correction (no flow, which would refill it within a step) grows it to about twice
its boxes (its pressure is its little energy's), past a split size set to 1.5 * k**3
for this test of the tear (the default 8 is not reached in so small a start), and volume_split tears it: universe.splits rises, two new cells
exist (a volume and a joint), ABAB holds, every cell is one piece, the contact table equals a recount from the
boxes, the pieces share the
torn volume's labels, and not one quantum is lost, also over 5 steps after.
"""
import numpy as np

from physics.potts import Settings, size_correction, stats, step, volume_split
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import cell_id, connected_everywhere, even_universe, table_matches_boxes


@needs_gpu
def test_a_starved_volume_tears_and_abab_and_energy_hold():
    universe = even_universe(6, 6)
    starved = cell_id(6, 2, 2, 2)
    fed = cell_id(6, 4, 4, 4)
    owner = universe.owner.numpy()
    energy = universe.energy.numpy()
    kept = energy[owner == starved] // 20
    moved = int((energy[owner == starved] - kept).sum())
    energy[owner == starved] = kept
    spread, rest = divmod(moved, int(np.sum(owner == fed)))
    energy[owner == fed] += spread
    energy[tuple(np.argwhere(owner == fed)[0])] += rest
    universe.energy.assign(energy)
    settings = Settings.all_on(split_boxes=1.5)
    start_total = int(energy.sum())
    start_cells = universe.cell_count
    size_correction(universe, settings)
    grown = int(universe.cell_boxes.numpy()[starved])
    torn = volume_split(universe, settings)
    end = stats(universe, settings)
    print(f"starved volume grew to {grown} boxes; splits {end['splits']}, refused {universe.scratch.get('split_refused', 0)}, cells {end['cells']}, "
          f"joints {end['joints']}, collapses {end['collapses']}")
    assert grown > settings.split_boxes * universe.base_boxes
    assert torn >= 1 and end["splits"] == torn
    assert universe.cell_count == start_cells + 2 * torn
    assert end["cells"] + end["joints"] == start_cells + 2 * torn
    assert end["abab_breaks"] == 0
    assert int(universe.energy.numpy().sum()) == start_total
    alive = np.flatnonzero(universe.alive.numpy() == 1)
    assert connected_everywhere(universe.owner.numpy(), alive)
    assert table_matches_boxes(universe)
    labels = universe.labels.numpy()
    assert abs(labels[starved] + labels[start_cells:start_cells + 2].sum() - 1.0) < 1e-5
    for _ in range(5):
        step(universe, settings)
    later = stats(universe, settings)
    assert later["abab_breaks"] == 0
    assert int(universe.energy.numpy().sum()) == start_total
