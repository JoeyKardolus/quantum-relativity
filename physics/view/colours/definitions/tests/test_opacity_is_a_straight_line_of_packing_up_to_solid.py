"""The opacity law and the palette, on numbers worked out by hand."""
import numpy as np

from physics.view.colours.definitions.opacity_law import opacity_law
from physics.view.colours.definitions.outline_opacity import outline_opacity
from physics.view.colours.definitions.palette import (JOINT_BLUE, OUTLINE, OUTLINE_FULL_AT, OUTLINE_INSET, SOLID,
                                                      VOLUME_RED, WHOLE_POWER)


def test_the_starts_mean_and_anything_thinner_is_clear_and_four_times_it_is_solid():
    assert SOLID == 4.0
    assert opacity_law(np.array([0.0, 0.4, 1.0, 1.75, 2.5, 3.25, 4.0, 900.0])).tolist() == [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_a_packing_that_is_no_number_or_below_nothing_is_clear():
    assert opacity_law(np.array([np.nan, -3.0])).tolist() == [0.0, 0.0]


def test_in_the_whole_grid_a_half_packed_cell_is_nearly_clear_and_a_lump_stays_solid():
    opacity = opacity_law(np.array([1.0, 2.5, 3.25, SOLID])) ** WHOLE_POWER
    assert opacity[0] == 0.0 and abs(opacity[1] - 0.0718) < 1e-3 and abs(opacity[2] - 0.335) < 1e-3 and opacity[3] == 1.0


def test_the_outline_is_always_there_up_close_and_fades_in_proportion_from_far():
    assert outline_opacity(OUTLINE_FULL_AT) == outline_opacity(500.0) == OUTLINE and 0.5 < OUTLINE <= 1.0
    assert outline_opacity(OUTLINE_FULL_AT / 4) == OUTLINE / 4 and outline_opacity(0.0) == 0.0 == outline_opacity(-1.0)
    assert 0 < OUTLINE_INSET < 0.25, "set in, yet a cell keeps most of its size"


def test_red_is_red_and_blue_is_blue():
    assert VOLUME_RED[0] > 4 * max(VOLUME_RED[1:]) and JOINT_BLUE[2] > 2 * max(JOINT_BLUE[:2])
