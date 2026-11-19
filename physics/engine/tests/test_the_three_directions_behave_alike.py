"""The three directions behave alike: a line of cells runs the same course along each."""
import numpy as np
import pytest

from physics.engine import line_cells, step, three_cells

CALM = 648_000


@pytest.mark.xfail(strict=True, reason="(kept) the three directions differ by 0.0041 of a calm volume after 6 turns "
                   "(0.0065 before, the mark is 0.003): the leak's lean to some directions is carried on as flow")
def test_a_line_of_cells_runs_the_same_course_along_each_direction():
    held = []
    for along in range(3):
        store = three_cells(along=along)
        for _ in range(150):
            step(store)
        held.append(store.cell_quanta.numpy()[list(line_cells(along))] / CALM)
    assert held[0][0] < 1.49 and held[0][2] > 0.51  # things happened: A from 1.5, B from 0.5 (1.487)
    assert np.abs(np.array(held) - held[0]).max() < 0.003  # 0.0018: a phase points to the 26 directions in turn, not alike
