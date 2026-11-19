"""The three directions behave alike: a line of cells runs the same course along each."""
import numpy as np

from physics.engine import line_cells, step, three_cells

CALM = 648_000


def test_a_line_of_cells_runs_the_same_course_along_each_direction():
    held = []
    for along in range(3):
        store = three_cells(along=along)
        for _ in range(150):
            step(store)
        held.append(store.cell_quanta.numpy()[list(line_cells(along))] / CALM)
    assert held[0][0] < 1.47 and held[0][2] > 0.53  # things happened: A from 1.5, B from 0.5 (1.446)
    assert np.abs(np.array(held) - held[0]).max() < 0.001  # measured: 0.00008
