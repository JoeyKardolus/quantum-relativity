"""Follow: axiom 7 after this step's collapses and tears.

Contract: `follow(state)` calls `carry_events` (this step's collapses and
tears move, share and react with halves) and then `settle` (every split reacted with,
named by a caller's detector, or out of momentum, settles whole in its pointer's cell).
It needs this step's `cell_quanta` (the step counts the cells first). `meet_fork` ran
earlier in the step, straight after `cross`; this is the step's one settling.
"""
from __future__ import annotations

from physics.engine.axiom_7_quantum.logic.carry_events import carry_events
from physics.engine.axiom_7_quantum.logic.settle import settle


def follow(state) -> None:
    """Carry the step's events into the splits and settle those whose time has come."""
    carry_events(state)
    settle(state)
