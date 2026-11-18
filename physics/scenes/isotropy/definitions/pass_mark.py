"""Pass mark of the isotropy scene, written 2026-11-17 before any run.

Known: space has no preferred direction.
Contract: `WITHIN` is 0.10. `passes(result)` takes the scene's result (48 places, one
volume packed 4 times, 2,000 steps, every rule on) and returns True when for every pair
the pull's ratio off-axis over on-axis is within WITHIN of the box law's ratio
(`pull_ratio / law_pull_ratio` in 0.9 .. 1.1) and the flow's ratio within WITHIN of 1.
A pair with no pull or no flow on its axis (ratio 0) does not pass. The plain diagonals
are reported and not judged.
"""
from __future__ import annotations

WITHIN = 0.10


def passes(result: dict) -> bool:
    """Whether pull and flow are the same off an axis as on it, at every pair, beside the box law."""
    return bool(result["pairs"]) and all(
        pair["law_pull_ratio"] != 0.0
        and abs(pair["pull_ratio"] / pair["law_pull_ratio"] - 1.0) <= WITHIN
        and abs(pair["flow_ratio"] / pair["law_flow_ratio"] - 1.0) <= WITHIN
        for pair in result["pairs"])
