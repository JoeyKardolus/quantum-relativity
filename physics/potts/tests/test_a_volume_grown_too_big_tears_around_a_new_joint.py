"""Volume split: an energy volume past the split size tears in two around a new joint (rule 8).

In an even start of 7**3 cells of 6**3 boxes one inner energy volume keeps a twentieth
of its energy (the rest moved into another volume), so it wants thousands of boxes;
the start's size correction (no flow, which would refill it within a step) grows it,
past a split size set to 1.5 * k**3 for this test of the tear (the default 8 is not
reached in so small a start), and volume_split tears it: universe.splits rises, two new
cells exist (a volume and a joint), every energy volume stays wrapped in joints, every
cell is one piece, the pieces share the torn volume's labels, and the total energy
holds to rounding, also over 5 steps after.
"""
import numpy as np

from physics.potts import Settings, size_correction, stats, step, volume_split
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import cell_id, connected_everywhere, even_universe, move_energy, wrapped_everywhere


@needs_gpu
def test_a_starved_volume_tears_and_stays_wrapped_and_energy_holds():
    universe = even_universe(7, 6)
    starved = cell_id(7, 3, 3, 3)
    fed = cell_id(7, 1, 1, 1)
    move_energy(universe, starved, fed, 0.05)
    settings = Settings.all_on(split_boxes=1.5)
    start_total = stats(universe, settings)["total_energy"]
    start_cells = universe.cell_count
    size_correction(universe, settings)
    grown = int(universe.cell_boxes.numpy()[starved])
    torn = volume_split(universe, settings)
    end = stats(universe, settings)
    print(f"starved volume grew to {grown} boxes; splits {end['splits']}, refused "
          f"{universe.scratch.get('split_refused', 0)}, cells {end['cells']}, joints {end['joints']}")
    assert grown > settings.split_boxes * universe.base_boxes
    assert torn >= 1 and end["splits"] == torn
    assert universe.cell_count == start_cells + 2 * torn
    assert end["cells"] + end["joints"] == start_cells + 2 * torn
    assert end["wrap_breaks"] == 0
    assert wrapped_everywhere(universe.owner.numpy(), universe.kind.numpy())
    assert abs(end["total_energy"] - start_total) < 1e-10 * start_total
    alive = np.flatnonzero(universe.alive.numpy() == 1)
    assert connected_everywhere(universe.owner.numpy(), alive)
    labels = universe.labels.numpy()
    assert abs(labels[starved] + labels[start_cells:start_cells + 2].sum() - 1.0) < 1e-5
    for _ in range(5):
        step(universe, settings)
    later = stats(universe, settings)
    assert later["wrap_breaks"] == 0
    assert abs(later["total_energy"] - start_total) < 1e-10 * start_total
