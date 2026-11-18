"""A seeded core run reads the same twice; a window over the whole run reads the whole run; a packed core's slow clock keeps its energy; bad windows are refused."""
import pytest

from physics.scenes.black_hole import black_hole, passes
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_one_core_read_whole_and_the_same_seed_read_twice():
    first = black_hole(16, energies=(8.0,), steps=4, window=4, every=1, settings=HELD)
    again = black_hole(16, energies=(8.0,), steps=4, window=4, every=1, settings=HELD)
    assert first["inflow"] == again["inflow"] and first["depth"] == again["depth"], "a seeded run reads the same twice"
    assert first["inflow"][0] == first["gathered"][0], "a window over the whole run is the whole run"
    print(f"8 times, 4 steps: core holds {first['core_energy'][0]:.3f} even volumes, gained {first['core_gain'][0]:+.4f}; "
          f"inflow per boundary {[round(flow, 4) for flow in first['inflow'][0]]}")
    assert abs(first["core_energy"][0] - 8.0) < 0.01, "a core 8 times as full has a clock so slow it hardly trades in 4 steps"
    assert first["core_clock"][0] < 0.1 * 4 * HELD.step_time, "its own time is under a tenth of the counter's"
    assert first["held_power"] != first["held_power"], "NaN: one energy gives no power"
    assert not passes(first)


def test_a_window_longer_than_the_run_a_tiny_grid_or_no_energy_is_refused():
    for arguments in ({"steps": 4, "window": 5}, {"steps": 4, "window": 0}, {"cells_per_side": 4, "steps": 2, "window": 1},
                      {"energies": (), "steps": 2, "window": 1}):
        with pytest.raises(ValueError):
            black_hole(**{"cells_per_side": 16, **arguments})
