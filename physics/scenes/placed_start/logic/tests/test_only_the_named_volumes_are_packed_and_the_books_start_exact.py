"""A placed start packs exactly the named volumes, sizes them by the size law and starts with clean books."""
import numpy as np
import pytest

from physics.engine import ENERGY_VOLUME, Settings, book_law, read_books, size_law_host
from physics.scenes.placed_start.logic.packed_start import packed_start
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_only_the_named_volumes_are_packed_and_four_times_the_energy_is_a_quarter_the_length():
    universe = packed_start(8, {(4, 4, 4): 4.0, (0, 2, 6): 2.0})
    held = universe.host("cell_quanta")
    assert held[292] == 4 * 648000 and held[22] == 2 * 648000
    volumes = np.flatnonzero(universe.host("kind") == ENERGY_VOLUME)
    rest = held[np.setdiff1d(volumes, [292, 22])]
    assert np.abs(rest / 648000.0 - 1.0).max() < 0.04, "every other volume is the even start's"
    length = universe.host("length")
    assert length[292] == size_law_host(4 * 648000, Settings().size_constant) == pytest.approx(2.5, rel=1e-6)
    books = read_books(universe)
    assert books["wrong_cells"] == 0 and books["wrong_boxes"] == 0
    assert set(book_law(universe.start_books, books).values()) == {0}


@needs_gpu
def test_a_joint_cannot_be_packed_and_a_factor_must_be_above_zero():
    with pytest.raises(ValueError):
        packed_start(8, {(1, 0, 0): 2.0})
    with pytest.raises(ValueError):
        packed_start(8, {(0, 0, 0): 0.0})
