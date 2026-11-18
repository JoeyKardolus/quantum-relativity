"""Going once round a row adds up every cell's length on it, and in an even grid the three rows through a cell are equally long within 1%."""
import numpy as np
import pytest

from physics.scenes.along_grid import log_slope, middle_distances, row_cells
from physics.scenes.placed_start import packed_start
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_a_whole_row_and_the_three_rows_of_an_even_grid():
    universe = packed_start(8, {})
    length = universe.host("length").astype(np.float64)
    rounds = []
    for axis in range(3):
        row = row_cells(8, 292, axis, 1, 7)
        assert len(set(row.tolist())) == 8, "a row of 8 places holds 8 different cells"
        once_round = middle_distances(universe, row)[-1] + 0.5 * length[row[-1]] + 0.5 * length[row[0]]
        assert once_round == pytest.approx(length[row].sum())
        rounds.append(once_round)
    assert max(rounds) / min(rounds) < 1.01
    assert rounds[0] == pytest.approx(4 * 10.0 + length[row[1::2]].sum(), rel=0.02), "four volumes of 10 and four joints as long as the balance makes them"
    steps = middle_distances(universe, row_cells(8, 292, 0, 1, 6))[::2]
    assert log_slope([1.0, 2.0, 3.0], steps[1:]) == pytest.approx(1.0, abs=0.02), "volumes lie evenly along an even row"
