"""Pass mark of the dimension scene, written 2026-11-17 before any run.

Known: 3.
Contract: `passes(result)` takes the scene's result (24 places a side, measured at the
start and after 5,000 steps) and returns True when the ball dimension and the spectral
dimension at the end are both within 10% of KNOWN_DIMENSION (physics.measure passes:
2.7 .. 3.3). The plain wrapping grid's own numbers (`reference`) are reported beside,
not judged.
"""
from __future__ import annotations

from physics.measure import passes as within_known
from physics.scenes.dimension.definitions.known import KNOWN_DIMENSION


def passes(result: dict) -> bool:
    """Whether both dimensions at the end are within 10% of 3."""
    end = result["end"]
    return bool(within_known(end["ball_dimension"], KNOWN_DIMENSION)
                and within_known(end["spectral_dimension"], KNOWN_DIMENSION))
