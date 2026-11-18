"""Pass mark of the Hubble gap scene, written 2026-11-17 before any run.

Known: late over early growth rate 73.5 / 67.4 = 1.09.
Contract: `LOWEST` 1.04 and `HIGHEST` 1.14 bound the ratio. `passes(result, stands)`
takes the scene's result (24 places a side, 5,000 steps) and whether the universe
stands at the end (the fit's end rule, judged outside this package) and returns True
when both rates are above 0, late over early lies in LOWEST .. HIGHEST and the universe
stands. Two shrinking rates never pass, whatever their ratio.
"""
from __future__ import annotations

LOWEST = 1.04
HIGHEST = 1.14


def passes(result: dict, stands: bool) -> bool:
    """Whether both rates are growth and their ratio is the measured gap's, in a standing universe."""
    growing = result["early_rate"] > 0.0 and result["late_rate"] > 0.0
    return bool(growing and LOWEST <= result["late_over_early"] <= HIGHEST and stands)
