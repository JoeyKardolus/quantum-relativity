"""Pass mark of the two lumps scene (gravity), written 2026-11-17 before any run.

Known: the pull falls as 1 / distance squared; two masses fall toward each other.
Contract: `FROM_VOLUMES` is 3, `POWER_WITHIN` 0.2 and `KEPT` 0.5. All three functions
take the scene's result (48 places, two volumes packed 4 times, 2, 3, 4, 6, 8 volumes
apart, 2,000 steps, every rule on).
  `falls_off(result)`  the pull's power against distance, over the distances of
        FROM_VOLUMES volumes and more, is within POWER_WITHIN of the 1 / d squared
        law's power measured the same way in the same box. NaN (a pull at or below 0,
        or under 2 such distances) does not pass.
  `falls(result)`  at every distance the two lumps' extra energy closed in by more than
        the scatter of a lump alone (the largest |closing_alone|), and more than KEPT
        of it is still in the two lump volumes at the end.
  `passes(result)`  both: gravity works only if both pass.
Changed 2026-11-18 with the rebuilt engine, before any run on it: the fall-off was
judged on a run with crossing off; the engine has no off switch any more, so it is
judged with every rule on.
"""
from __future__ import annotations

import math

from physics.scenes.along_grid.definitions.log_slope import log_slope

FROM_VOLUMES = 3
POWER_WITHIN = 0.2
KEPT = 0.5


def falls_off(result: dict) -> bool:
    """Whether the pull falls with distance as the 1 / d squared law does in the same box."""
    far = [index for index, gap in enumerate(result["apart"]) if gap >= FROM_VOLUMES]
    distance = [result["distance"][index] for index in far]
    measured = log_slope(distance, [result["pull"][index] for index in far])
    law = log_slope(distance, [result["newton"][index] for index in far])
    return bool(not math.isnan(measured) and not math.isnan(law) and abs(measured - law) <= POWER_WITHIN)


def falls(result: dict) -> bool:
    """Whether the lumps closed in at every distance by more than a lone lump's scatter, still holding their energy."""
    scatter = max(abs(shift) for shift in result["closing_alone"])
    return bool(all(shift > scatter for shift in result["closing"]) and all(share > KEPT for share in result["kept"]))


def passes(result: dict) -> bool:
    """Whether gravity works: the pull falls off as the law and the lumps fall."""
    return falls_off(result) and falls(result)
