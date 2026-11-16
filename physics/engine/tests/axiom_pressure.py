"""Axiom pressure (test helper): every cell's push as the engine's own cell laws compute it, without changing the state.

Contract: `axiom_pressure(state, settings)` counts each cell's quanta and boxes and runs
the engine's own cell laws (cell_count.py, cell_laws.py) on a shallow copy of the state,
so the state is not changed. Returns (pressure, size, amount) for universe 0 as numpy
arrays per cell id: the cell's push (float64; the pull is not in it; 0 for an energy
volume, which only pulls, and for an id with no cell), its size (state.size: an energy
volume's wanted size, a joint's swinging size) and its quanta. A positive pressure is a
push.
"""
from __future__ import annotations

import copy

from physics.engine import Settings, State, cell_count, cell_laws


def axiom_pressure(state: State, settings: Settings) -> tuple:
    """(pressure, size, quanta) per cell of universe 0 from the engine's own cell laws."""
    scratch = copy.copy(state)
    cell_count(scratch)
    cell_laws(scratch, settings)
    return scratch.cell_push[0].copy(), scratch.size[0].copy(), scratch.cell_quanta[0].copy()
