"""Quantum react: something reacts with whatever moving energy has a part in a cell, and that energy settles whole.

Contract: `quantum_react(state, cell, universe=None)` is a reaction placed by a scene (a
detector) in energy volume `cell` of one universe, or of every universe of the batch
when `universe` is None: every moving energy with a part in that cell settles whole in
the cell its pointer is on at that moment (quantum_settle.py), which is the reacting
cell with the chance of its share and another of its cells otherwise; never by which
half reacted. A cell holding no part of a moving energy reacts with nothing. Returns
the events (quantum_store.py), [] when nothing was there.
"""
from __future__ import annotations

from physics.engine.quantum_settle import quantum_settle
from physics.engine.quantum_store import quantum_store
from physics.engine.state import State


def quantum_react(state: State, cell: int, universe: int | None = None) -> list:
    """Settle every moving energy that has a part in `cell` (one universe, or all)."""
    if universe is not None:
        state.universe(universe)
    met = [record for record in quantum_store(state)["moving"]
           if cell in record["cells"] and universe in (None, record["universe"])]
    return quantum_settle(state, met, "reaction")
