"""Why ended: the marks at which a universe is over, so stepping it further measures nothing.

Contract: `why_ended(tears, start_volumes, alive, spent_share)` returns "" while the
universe stands, else which mark it passed, in words:
  torn_out                more than two tears per start volume: it tears without end
  ALIVE_FLOOR       0.05  fewer than this share of the start's volumes alive: it fell
                          into a few cells
  SPENT_CAP         0.95  more than this share of the size in overstretched volumes
A run ended at a mark is scored on its samples so far: an economy, since such universes
take seconds a step.
`NOTHING_LEFT` names a fourth end, which no measure can be taken of: no energy volume
holds any energy (collapses took more off the volumes than they held). Such a run scores 0.
"""
from __future__ import annotations

from physics.fit.search.definitions.torn_out import torn_out

ALIVE_FLOOR = 0.05
SPENT_CAP = 0.95
NOTHING_LEFT = "no energy volume holds any energy"


def why_ended(tears: int, start_volumes: int, alive: float, spent_share: float) -> str:
    """The mark the universe passed, in words; "" while it stands."""
    if torn_out(tears, start_volumes):
        return "more than two tears per start volume"
    if alive < ALIVE_FLOOR:
        return "fewer than 5% of the start's volumes alive"
    if spent_share > SPENT_CAP:
        return "more than 95% of the size spent"
    return ""
