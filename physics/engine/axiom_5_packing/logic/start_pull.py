"""Start pull: the start's energy has always pulled, so the pull starts at rest, not at nothing.

Contract: `start_pull(state, settings)` sets every clock to 0, the pull to 0 and then to
the pull at rest for the start's energy (`rest_pull`): no wave is switched on at step 1,
and at the start the pull is already deeper near more energy and weaker with distance.
Needs `length`. Writes `pull`, `pull_old`, `pull_new` (working copy), `clock`. Changes
no quantum.
"""
from __future__ import annotations

from physics.engine.axiom_5_packing.logic.rest_pull import rest_pull


def start_pull(state, settings) -> None:
    """Set every clock to 0 and the pull to the field at rest for the energy as it lies."""
    state.clock.zero_()
    state.pull.zero_()
    rest_pull(state, settings)
