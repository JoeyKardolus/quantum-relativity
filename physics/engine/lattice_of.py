"""Lattice of: one universe's forced cell lattice as host arrays, to start a new State on it.

Contract: `lattice_of(state, universe=0)` returns the dict State.from_host takes as
`lattice`: "cells" (real cells per axis), "owner" (int32 per box) and "corners"
(float32 lattice vertices) of that universe, copied from the GPU. A start built from
new_universe's quanta (more quanta in some boxes, a phase imprinted) keeps its cells
this way; each box's sign must still match its cell's kind. Returns None for a state
without a lattice. Raises ValueError for a universe the batch does not hold.
"""
from __future__ import annotations

from physics.engine.host_boxes import host_boxes
from physics.engine.state import State


def lattice_of(state: State, universe: int = 0) -> dict | None:
    """The universe's cells per axis, box owners and lattice corners, for State.from_host."""
    boxes = host_boxes(state, universe)
    if not any(boxes["cells"]):
        return None
    return {"cells": boxes["cells"], "owner": boxes["owner"], "corners": boxes["corners"]}
