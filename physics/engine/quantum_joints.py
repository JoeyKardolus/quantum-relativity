"""Quantum joints: an energy volume's joints as a moving energy remembers them, to notice a collapse or a tear.

Contract: `quantum_joints(state, cell)` returns {joint id: the energy volumes at the
other end of that joint's ways from `cell`, a sorted tuple, empty for a joint with no
way} for every joint beside energy volume `cell` now (state.neighbours, state.ways). A
cell's joints change only by a collapse or a tear (axiom 3), so a record that kept this
dict sees either by comparing (quantum.py).
"""
from __future__ import annotations

from physics.engine.state import State


def quantum_joints(state: State, cell: int) -> dict:
    """A volume's joints now, {joint: the volumes at the far ends of its ways from the cell}."""
    found = {}
    for joint in state.joints_of(cell):
        first, second = state.sides(joint)
        across = (set(second) if cell in first else set()) | (set(first) if cell in second else set())
        found[joint] = tuple(sorted(across - {cell}))
    return found
