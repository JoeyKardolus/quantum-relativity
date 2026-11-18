"""Pass mark of the timescape scene, written 2026-11-17 before any run.

Known: real supernovae give q about -0.55; timescape says that is the lumps' slow clocks.
Contract: `GREW` is 1.01 and `LOWER_BY` 0.05. `passes(result, stands)` takes the scene's
result (24 places a side, 5,000 steps, 8 observers) and whether the universe stands
(the fit's end rule, judged outside this package) and returns True when the size at the
end is GREW times the start's or more, q on the observers' clocks is below 0 and lower
than q on the counter by more than LOWER_BY, and the universe stands. A NaN q does not pass.
"""
from __future__ import annotations

GREW = 1.01
LOWER_BY = 0.05


def passes(result: dict, stands: bool) -> bool:
    """Whether growth looks sped up on the lumps' clocks, and more so than on the counter."""
    grew = result["size"][-1] / result["size"][0] >= GREW
    sped_up = result["q_observer"] < 0.0 and result["q_observer"] < result["q_counter"] - LOWER_BY
    return bool(grew and sped_up and stands)
