"""A snapshot keeps every cell's quanta and length, names the lump volumes, and counts an endless or gone cell as no length."""
import numpy as np

from physics.fit.measures.logic.take_snapshot import take_snapshot
from physics.fit.measures.logic.tests.hand_cells import lattice_cells


def test_the_snapshot_of_a_hand_made_grid():
    cells = lattice_cells()
    cells["held"][[0, 40]] = [200, 450]
    cells["held"][10], cells["length"][10] = 0, np.inf
    shot = take_snapshot(cells)
    assert shot["lumps"].tolist() == [0, 40] and shot["plus"] == 200 + 450 + 5 * 100
    assert shot["length"][10] == 0.0 and shot["length"][0] == 10.0 and shot["length"][1] == 30.0
    assert shot["count"] == 64 and shot["version"] == 0 and shot["ways"] is cells["ways"]
    assert np.array_equal(shot["held"], cells["held"])
