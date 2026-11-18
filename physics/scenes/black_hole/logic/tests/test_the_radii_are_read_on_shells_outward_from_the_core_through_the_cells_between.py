"""The scene measures the region round a packed core: shells counted outward, distances through the cells between, radii from the flow and the pull."""
import numpy as np

from physics.scenes.black_hole.logic.black_hole import black_hole
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_shells_distances_flow_and_depth_hang_together_for_two_cores():
    result = black_hole(16, energies=(4.0, 8.0), steps=6, window=4, every=2, settings=HELD)
    assert result["shells"] == 4 and result["level"] >= 0.0 and result["known_power"] == 1.0
    for run in range(2):
        distance, depth, inflow = result["distance"][run], result["depth"][run], result["inflow"][run]
        assert len(distance) == len(depth) == len(inflow) == 5
        assert distance[0] == 0.0 and np.all(np.diff(distance) > 0.0), "shells lie further and further out"
        assert distance[1] > 5.0, "to the next volume: half the core, the joint between, half a volume"
        assert depth[-1] == 0.0, "depth is read below the last shell"
        held = result["held_volumes"][run]
        assert 0 <= held <= 3 and all(flow > 0.0 for flow in inflow[1:held + 1])
        assert held == 3 or inflow[held + 1] <= 0.0, "the region stops at the first boundary energy leaves through"
        assert result["held_radius"][run] == (distance[held] if held else 0.0)
        assert result["held_limit"][run] == int(held == 3)
        assert 0.0 <= result["depth_radius"][run] <= distance[-1]
        assert result["core_gain"][run] == inflow[0], "the core's own skin is reported apart"
    assert result["core_energy"][1] > result["core_energy"][0]
    assert result["core_clock"][1] < result["core_clock"][0], "the fuller core's clock is the slower"
