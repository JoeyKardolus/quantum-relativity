"""Size follows energy: a wrapped energy volume given 1.5x energy gives up boxes, one given 0.6x claims them.

The pull is off so only size, push and the energy fluid act; the start references are
the even start's (every cell k**3 boxes). Checked after 2 steps.
"""
from physics.potts import Settings, step
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import box_energy, cell_id, even_universe, set_box_energy


@needs_gpu
def test_more_energy_means_fewer_boxes_and_less_energy_more():
    universe = even_universe(7, 4)
    fed = cell_id(7, 3, 3, 3)
    starved = cell_id(7, 1, 3, 3)
    owner = universe.owner.numpy()
    energy = box_energy(universe)
    energy[owner == fed] *= 1.5
    energy[owner == starved] *= 0.6
    set_box_energy(universe, energy)
    settings = Settings.all_on(attraction=0.0)
    for _ in range(2):
        step(universe, settings)
    boxes = universe.cell_boxes.numpy()
    start = universe.base_boxes
    print(f"fed cell {boxes[fed]} boxes, starved cell {boxes[starved]} boxes, start {start}")
    assert boxes[fed] < start
    assert boxes[starved] > start
