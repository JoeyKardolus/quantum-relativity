"""Attach ladder: keep the detail ladder's store beside the engine's boxes, switched every step.

Contract: `attach_ladder(state, settings=None)` builds a ladder store
(physics.ladder.new_store) over the state's whole stacked grid, (universes * side / 8,
side / 8, side / 8) blocks of 8**3 boxes, loads the boxes' quanta and phase into it as
FULL blocks (all-empty blocks EMPTY), sets `state.ladder` and returns the store. From
then on every step() ends with ladder_sync.py, which writes the changed blocks and runs
the ladder's once-a-step switch (FULL, COARSE, REUSED, EMPTY by quietness). `settings` is
a physics.ladder.LadderSettings (storage accuracy, never physics; default its
defaults). Raises ValueError when the side is not a multiple of 8 (new_universe always
rounds it up to one).
"""
from __future__ import annotations

from physics.engine.state import State
from physics.ladder import LadderSettings, load_region, new_store
from physics.ladder.settings import BLOCK


def attach_ladder(state: State, settings: LadderSettings | None = None):
    """A ladder store holding the state's boxes, kept as state.ladder and synced every step."""
    if state.side % BLOCK:
        raise ValueError(f"the ladder needs a side that is a multiple of {BLOCK}, got {state.side}")
    blocks = (state.shape[0] // BLOCK, state.side // BLOCK, state.side // BLOCK)
    store = new_store(blocks, settings or LadderSettings(), device=state.quanta.device)
    load_region(store, state.quanta.numpy(), state.phase.numpy())
    state.ladder = store
    return store
