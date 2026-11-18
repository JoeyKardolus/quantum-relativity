"""A placed lump starts as the fullest volume, with the even grid's pull still everywhere, and a step keeps the books."""
import numpy as np

from physics.engine import ENERGY_VOLUME, book_law, read_books, step
from physics.scenes.placed_start import packed_start
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_the_pull_at_the_start_is_the_even_twins_and_one_step_keeps_the_books():
    lump = packed_start(8, {(4, 4, 4): 4.0}, settings=HELD)
    even = packed_start(8, {}, settings=HELD)
    assert np.array_equal(lump.host("pull"), even.host("pull")), "the pull has not heard of the lump at step 0"
    held = np.where(lump.host("kind") == ENERGY_VOLUME, lump.host("cell_quanta"), 0)
    assert int(np.argmax(held)) == 292
    assert lump.host("rate")[292] < even.host("rate")[292], "more packed, slower clock"
    step(lump, HELD)
    assert set(book_law(lump.start_books, read_books(lump)).values()) == {0}
    assert not np.array_equal(lump.host("pull"), even.host("pull")), "after a step the lump's pull is on its way"
