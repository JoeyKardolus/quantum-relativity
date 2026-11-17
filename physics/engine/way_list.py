"""Way list: every way of the grid now, as three host arrays.

Contract: `way_list(state)` returns (joints, first, second), int64 arrays of equal
length: one row per way, the living joint that carries it and the two energy volumes
at its ends (state.ways). Pristine face joints are read from the lattice in one go,
joints with their own ways from their lists.
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import JOINT, State


def way_list(state: State) -> tuple:
    """All ways now: (joint, volume, volume) per row."""
    side = state.side
    places = np.arange(state.lattice, dtype=np.int64)
    index = np.stack([places // (side * side), (places // side) % side, places % side], axis=1)
    odd = index % 2
    face = (odd.sum(axis=1) == 1) & (state.kind_h[:state.lattice] == JOINT)
    if state.sides_over:
        face[[joint for joint in state.sides_over if joint < state.lattice]] = False
    joints = places[face]
    low, high = index[face] - odd[face], index[face] + odd[face]
    low, high = low % side, high % side
    rows = [joints]
    firsts = [(low[:, 0] * side + low[:, 1]) * side + low[:, 2]]
    seconds = [(high[:, 0] * side + high[:, 1]) * side + high[:, 2]]
    extra = [(joint, pair[0], pair[1]) for joint in state.sides_over for pair in state.ways(joint)]
    if extra:
        table = np.array(extra, dtype=np.int64)
        rows.append(table[:, 0])
        firsts.append(table[:, 1])
        seconds.append(table[:, 2])
    return np.concatenate(rows), np.concatenate(firsts), np.concatenate(seconds)
