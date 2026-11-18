"""Pass mark of the bound lump scene, written 2026-11-17 before any run.

Known: a bound thing keeps its size while space grows.
Contract: `KEEPS_WITHIN` is 0.10 and `REST_GROWS` 1.05. `passes(result, stands_short,
stands_long)` takes the scene's result (24 places a side, 5,000 steps) and whether the
universe stands at 5,000 and at 20,000 steps (the fit's end rule, judged outside this
package) and returns True when the fullest volume is a lump at the end, over the second
half of the run its length changes by less than KEEPS_WITHIN while the other volumes'
mean length grows by more than REST_GROWS, and the universe stands both times.
"""
from __future__ import annotations

from physics.scenes.bound_lump.definitions.second_half_change import second_half_change

KEEPS_WITHIN = 0.10
REST_GROWS = 1.05


def passes(result: dict, stands_short: bool, stands_long: bool) -> bool:
    """Whether a lump kept its length over the second half while the rest of space grew."""
    kept = abs(second_half_change(result["lump_length"]) - 1.0) < KEEPS_WITHIN
    grew = second_half_change(result["rest_length"]) > REST_GROWS
    return bool(result["is_lump"] and kept and grew and stands_short and stands_long)
