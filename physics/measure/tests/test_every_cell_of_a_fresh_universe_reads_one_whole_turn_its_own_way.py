"""Axiom 2 on the measure: the engine starts every cell, volume and joint, as one whole swirl
round its own axis, so every layer of every cell reads exactly the expected turns."""
import numpy as np

from physics.engine import BLOCK, ENERGY_VOLUME, JOINT, NO_CELL, new_universe, read_cells
from physics.measure import cell_turns, expected_turns


def test_all_six_layers_of_every_volume_and_joint_read_the_expected_turns(device):
    universe = new_universe(cells_per_side=8, seed=1, device=device)
    kinds = read_cells(universe)["kind"]
    axes, senses = universe.host("swirl_axis"), universe.host("swirl_sense")
    living = np.flatnonzero(kinds != NO_CELL)
    assert np.count_nonzero(kinds == ENERGY_VOLUME) == 64 and np.count_nonzero(kinds == JOINT) == 448
    assert set(axes[living].tolist()) == {0, 1, 2} and set(senses[living].tolist()) == {-1, 1}
    wrong = [int(cell) for cell in living
             if cell_turns(universe.block("quanta", int(cell)), universe.block("phase", int(cell)), int(axes[cell]))
             != [expected_turns(int(axes[cell]), int(senses[cell]))] * BLOCK]
    assert wrong == []
