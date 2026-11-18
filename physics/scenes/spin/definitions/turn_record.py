"""Turn record: what a run of winding readings says about one cell.

Contract: `turn_record(readings, expected)` takes one reading per step, in step order
from step 1 (each a list of whole turns, one per layer across the cell's swirl axis),
and the whole turns a cell that is still its swirl reads in every layer. Returns a dict:
  seen        every value read in any layer at any step, sorted
  held_steps  the number of steps at which every layer read exactly `expected`
  first_off   the first step (counted from 1) at which one did not; -1: never
"""
from __future__ import annotations


def turn_record(readings: list, expected: int) -> dict:
    """The values seen, the steps held and the first step off, from one cell's readings."""
    held = [all(turns == expected for turns in layers) for layers in readings]
    return {"seen": sorted({int(turns) for layers in readings for turns in layers}), "held_steps": sum(held),
            "first_off": held.index(False) + 1 if False in held else -1}
