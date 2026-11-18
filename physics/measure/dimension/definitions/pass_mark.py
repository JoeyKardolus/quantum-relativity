"""Pass mark: how close a dimension measure must come to a known dimension to be trusted.

Contract: `WITHIN` is the share of the known dimension a measured one may be off, 0.1
(written before any run). `passes(measured, known)` is True when
|measured - known| <= WITHIN * known. It judges a measure on a reference whose
dimension is known; it is not a verdict on the engine's grid. Raises ValueError for
known <= 0.
"""
from __future__ import annotations

WITHIN = 0.1


def passes(measured: float, known: float) -> bool:
    """Whether a measured dimension lies within the pass mark of the known one."""
    if known <= 0:
        raise ValueError(f"a known dimension is above 0, got {known}")
    return abs(measured - known) <= WITHIN * known
