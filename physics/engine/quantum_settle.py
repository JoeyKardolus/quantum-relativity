"""Quantum settle: moving energies settle whole, each in the cell its pointer is on, drawn there by the direct pull.

Contract: `quantum_settle(state, records, cause)` ends every record of `records`
(quantum_store.py) at once: all of its quanta in its other cells are moved into the
cell the pointer is on (quantum_move.py: whole quanta, any distance, within the call;
axiom 7: it settles whole in the cell the pointer is on, the direct pull between the
halves is private to them and does not pass through neighbouring cells), the record
leaves the store's moving list and an event with `cause` is logged. A whole energy
(one cell) comes to rest where it is, nothing drawn. The settled energy is ordinary
energy of that cell from then on. Returns the events, in the records' order.
"""
from __future__ import annotations

from physics.engine.quantum_move import quantum_move
from physics.engine.quantum_store import quantum_store
from physics.engine.state import State


def quantum_settle(state: State, records: list, cause: str) -> list:
    """Settle each record whole in its pointer's cell; log and return the events."""
    store = quantum_store(state)
    moves, events = [], []
    for record in records:
        home = record["cells"][record["pointer"]]
        drawn = 0
        for cell, part in zip(record["cells"], record["parts"]):
            if cell != home:
                moves.append((record["universe"], cell, home, part))
                drawn += part
        events.append({"serial": record["serial"], "universe": record["universe"], "step": int(state.step),
                       "cause": cause, "cell": home, "amount": record["amount"], "drawn": drawn,
                       "parts": dict(zip(record["cells"], record["parts"]))})
    quantum_move(state, moves)
    ended = {id(record) for record in records}
    store["moving"] = [record for record in store["moving"] if id(record) not in ended]
    store["ended"].extend(events)
    return events
