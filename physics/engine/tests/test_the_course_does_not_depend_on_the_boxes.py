"""The same scene runs the same course at 6, 12 and 24 boxes across: boxes are not space.

The course is counted in turns of a calm swirl, the cell's own time (a step is not).
"""
import numpy as np

from physics.engine import line_cells, step, three_cells


def _course(across: int) -> np.ndarray:
    """A's, the joint's and B's quanta in calm volumes after half a turn and after one turn."""
    store, seen = three_cells(across), []
    a_turn = 2 * np.pi / store.turn_per_step  # steps in one turn of a calm swirl
    for moment in (0.5, 1.0):
        while store.step < round(moment * a_turn):
            step(store)
        seen.append(store.cell_quanta.numpy()[list(line_cells())] / (3000 * across ** 3))
    return np.array(seen)


def test_the_course_is_the_same_at_6_12_and_24_boxes_across():
    least, more, most = _course(6), _course(12), _course(24)
    assert least[1, 0] < 1.15 and least[1, 1] < -1.1  # things happened: A from 1.5, joint from -1
    # within 0.03 of a calm volume's quanta (measured: 0.017 at most); before, 24 barely crossed
    assert np.abs(more - least).max() < 0.03 and np.abs(most - least).max() < 0.03
