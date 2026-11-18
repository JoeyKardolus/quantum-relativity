"""The opacity law and the palette, on numbers worked out by hand."""
import numpy as np

from physics.view.colours.definitions.opacity_law import opacity_law
from physics.view.colours.definitions.palette import JOINT_BLUE, SOLID, VOLUME_RED, WHOLE_POWER


def test_the_starts_mean_is_a_quarter_and_four_times_it_is_solid():
    assert SOLID == 4.0
    assert opacity_law(np.array([0.0, 0.4, 1.0, 2.0, 3.0, 4.0, 900.0])).tolist() == [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_a_packing_that_is_no_number_or_below_nothing_is_clear():
    assert opacity_law(np.array([np.nan, -3.0])).tolist() == [0.0, 0.0]


def test_in_the_whole_grid_the_even_start_is_nearly_clear_and_a_lump_stays_solid():
    opacity = opacity_law(np.array([1.0, 3.0, SOLID])) ** WHOLE_POWER
    assert abs(opacity[0] - 0.00515) < 1e-4 and abs(opacity[1] - 0.335) < 1e-3 and opacity[2] == 1.0
    through_32_places = 1.0 - (1.0 - opacity[0]) ** 64
    assert 0.1 < through_32_places < 0.4, "32 places of even cells still show what lies behind them"


def test_red_is_red_and_blue_is_blue():
    assert VOLUME_RED[0] > 4 * max(VOLUME_RED[1:]) and JOINT_BLUE[2] > 2 * max(JOINT_BLUE[:2])
