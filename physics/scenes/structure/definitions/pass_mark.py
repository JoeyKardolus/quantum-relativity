"""Pass mark of the structure scene, written 2026-11-17 before any run.

Known: lumps form over regions and make a web.
Contract: `LUMPS` is 5 and `REGION` 8. `passes(result, stands_short, stands_long)` takes
the scene's result (24 places a side, 5,000 steps) and whether the universe stands at
5,000 and at 20,000 steps (the fit's end rule, judged outside this package) and returns
True when the contrast between volumes is larger at the end than at the start, there
are LUMPS lumps or more, the largest joined region of lump volumes holds REGION or
more, and the universe stands both times. Threads of a web: no measure is built; that
part cannot pass and is reported as not measured.
"""
from __future__ import annotations

LUMPS = 5
REGION = 8


def passes(result: dict, stands_short: bool, stands_long: bool) -> bool:
    """Whether contrast grew into enough lumps joined over a region, in a universe that stood."""
    grew = result["rms_contrast"][-1] > result["rms_contrast"][0]
    lumped = result["lumps"][-1] >= LUMPS and result["largest_region"][-1] >= REGION
    return bool(grew and lumped and stands_short and stands_long)
