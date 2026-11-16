"""Quantum packet: name part of an energy volume's energy as one moving energy, tracked whole.

Contract: `quantum_packet(state, cell, quanta, universe=0)` says that `quanta` whole
quanta of energy volume `cell` are one moving energy (a packet on its way) and returns
its serial. No box is changed and no energy is made: the quanta are the cell's own,
energy has no identity, the record only keeps how much of each cell's energy belongs to
this moving energy (quantum_store.py). It starts whole in one cell, phase 0, every
quantum with its direction, no oscillation and no pointer to jump (those start at a
split, quantum_hand.py). Raises ValueError when the cell is not an energy volume of
that universe or `quanta` is below 1.
"""
from __future__ import annotations

import numpy as np

from physics.engine.quantum_joints import quantum_joints
from physics.engine.quantum_store import quantum_store
from physics.engine.state import State


def quantum_packet(state: State, cell: int, quanta: int, universe: int = 0) -> int:
    """Track `quanta` of an energy volume's energy as one moving energy; returns its serial."""
    state.universe(universe)
    if not 0 <= cell < state.capacity or state.kind[universe, cell] != 1:
        raise ValueError(f"cell {cell} of universe {universe} is not an energy volume")
    if quanta < 1:
        raise ValueError("a moving energy holds at least 1 quantum")
    store = quantum_store(state)
    serial = store["serial"]
    store["serial"] += 1
    store["moving"].append({
        "serial": serial, "universe": int(universe), "amount": int(quanta), "cells": [int(cell)],
        "parts": [int(quanta)], "phases": [0.0], "directed": int(quanta), "speed": None, "turn": 0.0, "jumps": 0,
        "pointer": 0, "born": None, "launched": int(state.step),
        "seen": int(state.collapses[universe] + state.splits[universe]),
        "joints": {int(cell): quantum_joints(state, universe, cell)},
        "rng": np.random.default_rng((int(state.seeds[universe]), serial, 7))})
    return serial
