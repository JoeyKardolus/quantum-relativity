"""Quantum store: the moving energies of a state that are tracked as one whole each, and the log of how they ended.

Contract: `quantum_store(state)` returns the state's store (state.quantum), made empty
on first use: {"moving": [record, ...], "ended": [event, ...], "serial": int}. Axiom 7
(bible physics/quantum-relativity, "The axioms"): moving energy can split over several
routes at once; every split gets a direct pull between its halves, private to them, and
its own oscillation with a pointer. A record is one moving energy, whole or split:
  serial    its number in this state, counted from 0
  universe  the universe it lives in
  amount    its whole quanta; they sit in the boxes of its cells like any other energy
  cells     the energy volumes holding part of it (cell ids); two or more: it is split
  parts     per cell the whole quanta of this energy there (sum = amount, each >= 1);
            a cell's share of the split energy is parts / amount
  phases    per cell the phase of this energy's wave there (radians): its fluid is
            sqrt(part) e^{i phase}, so its parts add and cancel where they meet
  directed  the quanta of it that still have their direction; friction takes it (quantum.py)
  speed     the split's own oscillation: turns per unit of the counter, set at the
            split from the record's own draws, None while whole. Nothing reads it but
            quantum.py; no measure of the engine reports it (nobody inside can know it)
  turn      turns made since the split, jumps the pointer's jumps so far
  pointer   index into cells: the cell the pointer is on
  born      the step of the split (None while whole), launched the step it was named
  seen, joints  the universe's collapses + volume splits last looked at, and per cell
            its joints then, {joint: the volume on its other side} (quantum.py)
  rng       the record's own draws, numpy Generator seeded by (universe seed, serial)
An event is {"serial", "universe", "step", "cause", "cell", "amount", "drawn", "parts"}:
cause "reaction", "momentum", "collapse or tear" (it settled whole in `cell`, `drawn`
quanta pulled in from its other cells by the direct pull; `parts` what each cell held
just before) or "merge" (a handover brought a split back together in `cell`, nothing
drawn; that energy moves on whole).
"""
from __future__ import annotations

from physics.engine.state import State


def quantum_store(state: State) -> dict:
    """The state's moving energies and their endings, made empty on first use."""
    if state.quantum is None:
        state.quantum = {"moving": [], "ended": [], "serial": 0}
    return state.quantum
