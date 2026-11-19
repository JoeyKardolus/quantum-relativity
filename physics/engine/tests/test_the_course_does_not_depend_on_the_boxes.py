"""The same scene runs the same course at 6, 12 and 24 boxes across, counted in turns of a calm swirl: boxes are not space."""
import numpy as np

from physics.engine import line_cells, step, three_cells


def _course(across: int, parts: tuple) -> np.ndarray:
    """A's, the joint's and B's quanta in calm volumes after half a turn and after one turn."""
    store, seen = three_cells(across, parts=parts), []
    a_turn = 2 * np.pi / store.turn_per_step  # steps in one turn of a calm swirl
    for moment in (0.5, 1.0):
        while store.step < round(moment * a_turn):
            step(store)
        seen.append(store.cell_quanta.numpy()[list(line_cells())] / (3000 * across ** 3))
    return np.array(seen)


def test_the_course_is_the_same_at_6_12_and_24_boxes_across():
    for parts in ((3, -2, 1), (4, -2, 4)):  # energy crossing; then two full volumes squeezing their joint
        least, more, most = (_course(across, parts) for across in (6, 12, 24))
        print(parts, least[1], np.abs(more - least).max(), np.abs(most - least).max())
        assert least[1, 0] < 1.495 or least[1, 1] < -1.2  # things happened: A from 1.5; the joint from -1
        # within 0.01 of a calm volume's quanta (measured: 0.002 crossing, 0.006 squeezing, in a
        # turn; the crossing drifts apart later, 0.023 after four turns: wider caps carry more)
        assert np.abs(more - least).max() < 0.01 and np.abs(most - least).max() < 0.01
