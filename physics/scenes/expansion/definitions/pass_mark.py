"""Pass mark of the expansion shape scene, written 2026-11-17 before any run.

Known: the real universe grew, first slowing, then speeding up.
Contract: `GROWTH` is 1.05. `passes(result, stands_short, stands_long)` takes the
expansion scene's result (run at 24 places a side for 5,000 steps) and whether the
universe still stands at 5,000 and at 20,000 steps (the fit's end rule, judged outside
this package), and returns True when the size at the end is GROWTH times the start's or
more, the growth rate's trend is below 0 over the first half and above 0 over the
second, and the universe stands both times. A NaN trend does not pass.
"""
from __future__ import annotations

GROWTH = 1.05


def passes(result: dict, stands_short: bool, stands_long: bool) -> bool:
    """Whether an expansion result grew enough, slowed then sped up, and its universe stood."""
    grew = result["size_growth"][-1] >= GROWTH
    turned = result["size_rate_trend_early"] < 0.0 < result["size_rate_trend_late"]
    return bool(grew and turned and stands_short and stands_long)
