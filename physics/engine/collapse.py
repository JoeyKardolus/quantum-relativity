"""Collapse: a joint squeezed too far gives way; its energy is taken off the rest and its volumes become one cell.

Contract: `collapse(state, held, joint)` collapses one living joint that carries a way
(the two energy volumes it lies between; a joint a tear left between more, every one of
them) and returns True, or returns False and does nothing when it is no such joint.
`held` is the host's copy of the per-cell arrays (events.py: cell_quanta, pull,
pull_old, clock, jiggle), changed here and written back by the caller. Axiom 6:
  - the joint is gone: its id is no cell any more, its boxes are emptied, its minus
    quanta leave the total (state.taken_joint), and every neighbour drops it
  - its volumes become one cell at once, the lowest id of them: energy has no place, so
    the other volume's quanta are poured box by box onto the same boxes of the kept
    block (their waves add: phase arg(sqrt(n) e^{i a} + sqrt(m) e^{i b})) and its id is
    emptied (state.fell_into[other] = kept). The kept cell takes over the other's
    neighbours in their directions and every neighbour's entry for the other; on every
    joint's sides the other is now the kept cell, and a way left between the cell and
    itself is no way
  - the joint's energy is taken off the merged volume's plus quanta, the same number
    (at most all it holds above 1 a box; state.taken_volumes), each box giving in
    proportion to what it holds above 1, exactly
The pattern stays A B A B: the merged cell's neighbours are the two volumes' joints, so
it is still wrapped by joints on every side and touches no volume, and the web loses one
joint whose own neighbours all still touch the merged cell. Raises OverflowError when a
merged box would pass 2 ** 30 quanta. Its pull, clock and heat
are the two cells' quanta-weighted means. Counts state.collapses, state.merges (volumes
that fell into another) and state.version.
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import BLOCK, BOXES, ENERGY_VOLUME, JOINT, State


def collapse(state: State, held: dict, joint: int) -> bool:
    """Collapse one joint: take its energy off the rest and make its volumes one cell."""
    joint = int(joint)
    if state.kind_h[joint] != JOINT:
        return False
    first, second = state.sides(joint)
    volumes = sorted(set(first) | set(second))
    if not first or not second or len(volumes) < 2:
        return False
    taken = int(held["cell_quanta"][joint])
    _drop(state, held, joint)
    for other in volumes[1:]:
        _merge(state, held, volumes[0], other)
    _take(state, held, volumes[0], taken)
    state.taken_joint += taken
    state.collapses += 1
    state.version += 1
    return True


def _drop(state: State, held: dict, joint: int) -> None:
    """Remove a joint from the grid: every neighbour forgets it, its boxes are emptied."""
    ids, _ = state.neighbours(joint)
    for other in sorted({int(one) for one in ids}):
        theirs, codes = state.neighbours(other)
        keep = theirs != joint
        state.set_neighbours(other, theirs[keep], codes[keep])
    _empty(state, held, joint)


def _empty(state: State, held: dict, cell: int) -> None:
    """Make an id no cell: no kind, no relations, no quanta in its boxes."""
    if state.kind_h[cell] == JOINT:
        state.set_sides(cell, [], [])
    state.adj_over.pop(cell, None)
    state.sides_over.pop(cell, None)
    state.kind_h[cell] = 0
    state.ways_h[cell] = 0
    state.set_block(cell, np.zeros((BLOCK, BLOCK, BLOCK), dtype=np.int32),
                    np.zeros((BLOCK, BLOCK, BLOCK), dtype=np.float32))
    held["cell_quanta"][cell] = 0


def _merge(state: State, held: dict, kept: int, other: int) -> None:
    """Pour volume `other` into volume `kept` and hand over its relations."""
    near, near_phase = state.block(kept)
    far, far_phase = state.block(other)
    wave = np.sqrt(near.astype(np.float64)) * np.exp(1j * near_phase) + np.sqrt(far.astype(np.float64)) * np.exp(
        1j * far_phase)
    if int(near.max()) + int(far.max()) >= 2 ** 30:
        raise OverflowError(f"volumes {kept} and {other} together hold more than a box can count (2**30 quanta)")
    state.set_block(kept, near + far, np.angle(wave))
    mine, my_codes = state.neighbours(kept)
    theirs, their_codes = state.neighbours(other)
    state.set_neighbours(kept, np.concatenate([mine, theirs]), np.concatenate([my_codes, their_codes]))
    for beside in sorted({int(one) for one in theirs}):
        ids, codes = state.neighbours(beside)
        state.set_neighbours(beside, np.where(ids == other, kept, ids), codes)
        first, second = state.sides(beside)
        if other in first or other in second:
            state.set_sides(beside, [kept if volume == other else volume for volume in first],
                            [kept if volume == other else volume for volume in second])
    mine_held, their_held = int(held["cell_quanta"][kept]), int(held["cell_quanta"][other])
    for name in ("pull", "pull_old", "clock", "jiggle"):
        held[name][kept] = (mine_held * float(held[name][kept])
                            + their_held * float(held[name][other])) / (mine_held + their_held)
    _empty(state, held, other)
    held["cell_quanta"][kept] = mine_held + their_held
    state.fell_into[other] = kept
    state.merges += 1


def _take(state: State, held: dict, volume: int, amount: int) -> None:
    """Take `amount` plus quanta off a volume's boxes, each in proportion to what it holds above 1, exactly."""
    quanta, _ = state.block(volume)
    spare = [int(value) - 1 for value in quanta.reshape(-1)]
    room = sum(spare)
    amount = min(amount, room)
    if amount > 0:
        given, run, parts = 0, 0, []
        for value in spare:
            run += value
            parts.append(amount * run // room - given)
            given += parts[-1]
        state.set_block(volume, quanta - np.array(parts, dtype=np.int32).reshape(quanta.shape))
        held["cell_quanta"][volume] -= amount
        state.taken_volumes += amount
    assert state.kind_h[volume] == ENERGY_VOLUME and held["cell_quanta"][volume] >= BOXES
