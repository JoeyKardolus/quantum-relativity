"""The marks, the relative packing and the colours, on plain arrays."""
import numpy as np
import pytest

from physics.view.colours.definitions.palette import JOINT_BLUE, SOLID, VOLUME_RED
from physics.view.colours.logic.cell_colours import cell_colours
from physics.view.colours.logic.packed_cells import packed_cells
from physics.view.colours.logic.relative_packing import relative_packing
from physics.view.colours.logic.start_marks import start_marks


def test_the_marks_are_the_starts_mean_packing_of_each_kind():
    kind = np.array([1, 1, -1, -1, -1, 0])
    packing = np.array([600.0, 800.0, 0.001, 0.002, 0.003, 55.0])
    marks = start_marks(kind, packing)
    assert marks == {1: 700.0, -1: pytest.approx(0.002)}


def test_a_start_with_nothing_to_read_against_is_refused():
    with pytest.raises(ValueError):
        start_marks(np.array([1, 1]), np.array([5.0, 5.0]))
    with pytest.raises(ValueError):
        start_marks(np.array([1, -1]), np.array([5.0, 0.0]))


def test_a_packing_is_read_against_its_own_kinds_mark_and_no_cell_is_nothing():
    relative = relative_packing(np.array([1, -1, 0, 1]), np.array([1400.0, 0.001, 9.0, 0.0]), {1: 700.0, -1: 0.002})
    assert relative.tolist() == [2.0, 0.5, 0.0, 0.0]


def test_kind_sets_the_hue_and_packing_only_the_opacity():
    packing = np.array([0.05, 1.0, 1.3, 2.0, 3.0, SOLID])
    red = cell_colours(np.ones(6, dtype=int), packing).astype(int)
    blue = cell_colours(-np.ones(6, dtype=int), packing).astype(int)
    assert (red[:, :3] == VOLUME_RED).all() and (blue[:, :3] == JOINT_BLUE).all(), "the hue never moves with packing"
    assert red[:, 3].tolist() == [0, 0, 26, 85, 170, 255] and np.array_equal(red[:, 3], blue[:, 3])
    assert cell_colours(np.array([1]), np.array([0.0]))[0, 3] == 0, "an emptied cell is fully transparent"
    assert cell_colours(np.array([1]), np.array([0.0])).dtype == np.uint8


def test_the_scale_is_fixed_and_clipped_never_stretched_to_the_frame():
    alone = cell_colours(np.array([1]), np.array([1.3]))
    among = cell_colours(np.array([1, 1, -1, 1]), np.array([1.3, 900.0, 0.001, 1e-9]))
    assert np.array_equal(alone[0], among[0])
    assert among[1].tolist() == [*VOLUME_RED, 255], "past the scale's top it stays solid"
    assert among[2].tolist() == [*JOINT_BLUE, 0] and among[3, 3] == 0


def test_the_whole_grid_shows_only_cells_whose_fill_would_be_seen_there():
    kind = np.array([1, 1, 1, -1, -1, 0])
    packing = np.array([700.0, 1100.0, 1300.0, 0.002, 0.008, 5000.0])
    assert packed_cells(kind, packing, {1: 700.0, -1: 0.002}).tolist() == [False, False, True, False, True, False], \
        "about 1.7 times the mean is the first step of 255 on the whole grid's scale; an id that is no cell never shows"
