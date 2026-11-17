"""The black hole scene measures the affected region round a packed core: shells counted outward from it, distances through the cells between, and radii that come from the flow and the pull, never from the core's own size."""
import numpy as np
import pytest

from physics.scenes import black_hole
from physics.scenes.black_hole import _box_law, _power
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_the_radii_are_read_on_shells_outward_from_the_core_through_the_cells_between():
    result = black_hole(16, energies=(4.0, 8.0), steps=60, window=20, every=5)
    assert result["shells"] == 4 and result["level"] > 0.0
    for run in range(2):
        distance, depth, inflow = result["distance"][run], result["depth"][run], result["inflow"][run]
        assert len(distance) == len(depth) == len(inflow) == 5
        assert distance[0] == 0.0 and np.all(np.diff(distance) > 0.0), "shells lie further and further out"
        assert distance[1] < 11.0, "a packed core is shorter, so its first shell is nearer than in an even grid"
        assert depth[-1] == 0.0, "depth is read below the half-box volume"
        held = result["held_volumes"][run]
        assert 0 <= held <= 3 and all(flow > 0.0 for flow in inflow[1:held + 1])
        assert held == 3 or inflow[held + 1] <= 0.0, "the region stops at the first boundary energy leaves through"
        assert result["held_radius"][run] == (distance[held] if held else 0.0)
        assert result["held_limit"][run] == int(held == 3)
        assert 0.0 <= result["depth_radius"][run] <= distance[-1]
        assert result["core_gain"][run] == inflow[0], "the core's own skin is reported apart"
    assert result["core_energy"][1] > result["core_energy"][0]


def test_the_box_law_is_one_over_distance_near_the_core_and_a_power_is_a_slope():
    near = [_box_law(radius, 500.0, 1000.0) for radius in (1.0, 2.0, 4.0)]
    assert near[0] / near[1] == pytest.approx(2.0, rel=0.02) and near[1] / near[2] == pytest.approx(2.0, rel=0.02)
    assert 0.0 < _box_law(400.0, 500.0, 1000.0) < 0.8 * (1.0 / 400.0 - 1.0 / 500.0), "near the half box the copies flatten the pull"
    assert _power([1.0, 2.0, 4.0], [3.0, 6.0, 12.0]) == pytest.approx(1.0)
    assert np.isnan(_power([2.0], [1.0]))
