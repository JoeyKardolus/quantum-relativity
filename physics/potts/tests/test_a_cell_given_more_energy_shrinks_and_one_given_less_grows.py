"""Size follows energy: in an even checkerboard, a cell given 1.5x energy gives up boxes, one given 0.6x claims them.

The pull is off so only size, push and flow act; the start references are the even
start's. Checked after 2 steps: the flow evens the two cells' energy out through
their joints within about 3 steps (measured: fed 96 -> 64 energy by step 3, starved
38 -> 61), after which their sizes follow the evened energy, not the start's.
"""
import numpy as np

from physics.potts import Settings, step
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import cell_id, even_universe


@needs_gpu
def test_more_energy_means_fewer_boxes_and_less_energy_more():
    universe = even_universe(6, 4)
    fed = cell_id(6, 2, 2, 2)
    starved = cell_id(6, 3, 3, 2)
    owner = universe.owner.numpy()
    energy = universe.energy.numpy()
    energy[owner == fed] = (energy[owner == fed] * 1.5).astype(np.int64)
    energy[owner == starved] = (energy[owner == starved] * 0.6).astype(np.int64)
    universe.energy.assign(energy)
    settings = Settings.all_on(attraction=0.0)
    for _ in range(2):
        step(universe, settings)
    boxes = universe.cell_boxes.numpy()
    start = universe.base_boxes
    print(f"fed cell {boxes[fed]} boxes, starved cell {boxes[starved]} boxes, start {start}")
    assert boxes[fed] < start
    assert boxes[starved] > start
