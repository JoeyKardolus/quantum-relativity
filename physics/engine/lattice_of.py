"""Lattice of: one universe's cell graph as host arrays, to start a new State on it.

Contract: `lattice_of(state, universe=0)` returns the dict State.from_host takes as
`lattice`: "owner" (int32 per box), "kind", "between", "place", "size", "speed",
"cell_clock" per cell id and "size_energy", copied from the GPU. A start built from
new_universe's boxes (more quanta in some boxes, a phase imprinted) keeps its cells
this way; each box's sign must still match its cell's kind. Pass wrap=state.wrap to
from_host with it. Raises ValueError for a universe the batch does not hold.
"""
from __future__ import annotations

from physics.engine.host_boxes import host_boxes
from physics.engine.state import State


def lattice_of(state: State, universe: int = 0) -> dict:
    """The universe's box owners and cell graph, for State.from_host."""
    boxes = host_boxes(state, universe)
    return {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "speed", "cell_clock",
                                           "size_energy")}
