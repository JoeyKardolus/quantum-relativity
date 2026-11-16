"""Quantum joints: an energy volume's joints as a moving energy remembers them, to notice a collapse or a tear.

Contract: `quantum_joints(state, universe, cell)` returns {joint id: the energy volume on
the joint's other side, -1 for none} for every joint of energy volume `cell` now
(state.between). A cell's joints change only by a joint collapse or a volume split
(bible rule 2), so a record that kept this dict sees either by comparing (quantum.py).
"""
from __future__ import annotations

from physics.engine.state import State


def quantum_joints(state: State, universe: int, cell: int) -> dict:
    """A volume's joints now, {joint: the volume on its other side (-1: none)}."""
    sides = state.between[universe]
    return {int(joint): int(sides[joint, 0] if sides[joint, 1] == cell else sides[joint, 1])
            for joint in state.joints_of(universe, cell)}
