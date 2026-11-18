"""React: a caller's detector reacts with whatever split has a half in one cell.

Contract: `react(state, cell)` adds `cell` to `state.quantum["detected"]`. It writes
nothing else: at the next `follow` every split with a half in that cell settles, and the
set is emptied. Inside the engine a split is reacted with by a collapse or a tear in a half's cell
(`carry_events`); this is the same from outside, for a scene's detector.
"""
from __future__ import annotations


def react(state, cell: int) -> None:
    """Have a detector react in `cell` at the next settling."""
    state.quantum["detected"].add(int(cell))
