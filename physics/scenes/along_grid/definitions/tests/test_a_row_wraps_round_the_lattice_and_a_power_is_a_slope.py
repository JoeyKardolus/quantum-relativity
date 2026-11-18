"""A row's cells are the next ids along its axis, wrapping; a power law's power is its log slope."""
import math

import pytest

from physics.scenes.along_grid.definitions.log_slope import log_slope
from physics.scenes.along_grid.definitions.row_cells import row_cells


def test_a_row_along_the_first_axis_wraps_from_the_last_place_to_the_first():
    core = (4 * 8 + 4) * 8 + 4
    assert row_cells(8, core, 0, 1, 5).tolist() == [core, core + 64, core + 128, core + 192, core - 256, core - 192]
    assert row_cells(8, core, 2, -1, 5).tolist() == [core, core - 1, core - 2, core - 3, core - 4, core + 3]
    assert row_cells(8, core, 1, 1, 0).tolist() == [core]
    with pytest.raises(ValueError):
        row_cells(8, core, 3, 1, 2)


def test_a_slope_recovers_the_power_and_is_nan_where_no_power_can_be_read():
    assert log_slope([1.0, 2.0, 4.0], [3.0, 6.0, 12.0]) == pytest.approx(1.0)
    assert log_slope([1.0, 2.0, 4.0], [1.0, 0.25, 0.0625]) == pytest.approx(-2.0)
    assert math.isnan(log_slope([2.0], [1.0])) and math.isnan(log_slope([1.0, 2.0], [1.0, -1.0]))
    with pytest.raises(ValueError):
        log_slope([1.0, 2.0], [1.0])
