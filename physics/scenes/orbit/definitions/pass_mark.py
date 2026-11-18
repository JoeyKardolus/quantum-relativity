"""Pass mark of the orbit scene (Kepler), written 2026-11-17 before any run.

Known: period squared over distance cubed is the same at every distance.
Contract: `AGREE_WITHIN` is 1.10. `passes(result)` takes the scene's result (32 places,
a big volume of 8 times and a small one of 2 times at 2, 3 and 4 volumes, 2,000 steps)
and returns True when the small one's extra energy went at least one whole turn round
the big one at two or more distances before half of it was lost, and period squared
over radius cubed agrees between those distances within 10% (largest over smallest at
most AGREE_WITHIN). No whole turn anywhere is a miss.
"""
from __future__ import annotations

AGREE_WITHIN = 1.10


def passes(result: dict) -> bool:
    """Whether two or more distances made a whole turn with one Kepler ratio between them."""
    whole = [run["period_squared_over_radius_cubed"] for run in result["runs"] if abs(run["turned"]) >= 1.0]
    return bool(len(whole) >= 2 and min(whole) > 0.0 and max(whole) / min(whole) <= AGREE_WITHIN)
