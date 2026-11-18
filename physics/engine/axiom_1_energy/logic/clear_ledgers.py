"""Clear ledgers: a step starts with empty counts of what happened in a step.

Contract: `clear_ledgers(state)` zeroes every array of `state.arrays` whose `each_step`
is True, of every axiom (what moved, what crossed, what was made, the press, the flags:
each is "this step's" and nothing else), and sets `state.changed = []`, this step's
events. It touches no other array and no count kept since the start (`made_total`,
`taken_joint`, ...): those are the books and are never cleared.
"""
from __future__ import annotations


def clear_ledgers(state) -> None:
    """Zero every per-step array and empty the list of this step's events."""
    for name, (dtype, shape, each_step) in state.arrays.items():
        if each_step:
            getattr(state, name).zero_()
    state.changed = []
