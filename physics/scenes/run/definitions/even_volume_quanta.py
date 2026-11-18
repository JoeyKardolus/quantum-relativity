"""Even volume quanta: what an energy volume holds at the start when every volume is alike.

Contract: `even_volume_quanta(settings)` returns `settings.volume_quanta` times the
boxes of a cell's block (physics.engine BOXES, 216) as a float: the quanta a lump is
measured against ("4 times" is 4 times this).
"""
from __future__ import annotations

from physics.engine import BOXES, Settings


def even_volume_quanta(settings: Settings) -> float:
    """The quanta of an even energy volume at the start."""
    return float(settings.volume_quanta) * BOXES
