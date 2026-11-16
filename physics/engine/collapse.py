"""Collapse: a joint squeezed past its limit gives way, and its two energy volumes become one cell at once.

Contract: `collapse(state, settings, squeeze)` takes each joint's squeeze (cell_laws.py,
its wanted length over its length minus 1) and collapses, most squeezed first, every
joint that lies between two energy volumes with squeeze >= collapse_squeeze (owner,
2026-11-16: when a joint collapses its two volumes become one cell at once). Joint J
between volumes A and B (A the lower id, it stays): every box of J turns plus, its
minus quanta becoming plus quanta of A, the same amount (bible rule 4: joint collapse
is the one place where pushing energy becomes pulling energy), marks bit 1; except a
box of J that has another volume's plus box or empty space or a wall among its 26
neighbours, or a joint other than A's and B's on a face: that box keeps its minus
quanta and goes to a neighbouring joint of A or B (marks bit 2), so the merged volume
stays wrapped on all 26 sides. Every box of B joins A (bit 2): B and J are no more. A
joint B had with a volume X becomes A's; where A and B both had a joint with X the two
become one joint (the higher id's boxes join the lower's, bit 2, sizes and speeds
added), so two neighbouring volumes still share exactly one joint. A's clock is the box
weighted mean of A's and B's. tallies[u, 1] counts the boxes turned plus, tallies[u, 3]
their quanta, state.collapses the joints collapsed. The boxes are changed on the host
and copied back. Returns the number of collapses.
"""
from __future__ import annotations

import numpy as np

from physics.engine.near import near
from physics.engine.settings import Settings
from physics.engine.state import State


def collapse(state: State, settings: Settings, squeeze: np.ndarray) -> int:
    """Collapse every joint squeezed past the limit; its two volumes become one."""
    ready = (state.kind == -1) & (squeeze >= settings.collapse_squeeze) & (state.between.min(axis=-1) >= 0)
    if not ready.any():
        return 0
    quanta, owner, marks = state.quanta.numpy(), state.owner.numpy(), state.marks.numpy()
    tallies = state.tallies.numpy()
    done = 0
    for universe in np.flatnonzero(ready.any(axis=1)):
        rows = state.universe(int(universe))
        for joint in sorted(np.flatnonzero(ready[universe]), key=lambda cell: -squeeze[universe, cell]):
            if state.kind[universe, joint] != -1 or state.between[universe, joint].min() < 0:
                continue
            flipped, amount = _fall_in(state, int(universe), int(joint), quanta[rows], owner[rows], marks[rows])
            tallies[universe, 1] += flipped
            tallies[universe, 3] += amount
            state.collapses[universe] += 1
            done += 1
    for name, values in (("quanta", quanta), ("owner", owner), ("marks", marks), ("tallies", tallies)):
        getattr(state, name).assign(values)
    return done


def _fall_in(state: State, universe: int, joint: int, quanta: np.ndarray, owner: np.ndarray,
             marks: np.ndarray) -> tuple:
    """Turn joint `joint` into its lower volume, B into A; returns (boxes turned plus, quanta turned plus)."""
    keep, gone = (int(cell) for cell in sorted(state.between[universe, joint]))
    theirs = np.concatenate([state.joints_of(universe, keep), state.joints_of(universe, gone)])
    boxes = owner == joint
    plus = quanta > 0
    foreign = plus & (owner != keep) & (owner != gone)
    blocked = near(foreign, state.wrap) | near(quanta == 0, state.wrap, wall=True)
    blocked |= near((quanta < 0) & ~np.isin(owner, theirs), state.wrap, faces_only=True)
    turned = boxes & ~blocked
    _rehome(state, owner, marks, boxes & blocked, theirs, joint)
    amount = int(-quanta[turned].astype(np.int64).sum())
    quanta[turned] = -quanta[turned]
    owner[turned] = keep
    marks[turned] |= 1
    moved = owner == gone
    owner[moved] = keep
    marks[moved] |= 2
    weights = np.array([state.cell_boxes[universe, keep], state.cell_boxes[universe, gone]], dtype=np.float64) + 1e-9
    clocks = state.cell_clock[universe, [keep, gone]]
    state.cell_clock[universe, keep] = float((weights * clocks).sum() / weights.sum())
    state.kind[universe, [joint, gone]] = 0
    state.size[universe, [joint, gone]] = 0.0
    state.speed[universe, [joint, gone]] = 0.0
    sides = state.between[universe]
    sides[sides == gone] = keep
    sides[joint] = -1
    state.between[universe] = np.sort(np.where(sides < 0, np.iinfo(np.int32).max, sides), axis=1)
    state.between[universe][state.between[universe] == np.iinfo(np.int32).max] = -1
    _join_twins(state, universe, keep, owner, marks)
    return int(turned.sum()), amount


def _rehome(state: State, owner: np.ndarray, marks: np.ndarray, kept: np.ndarray, theirs: np.ndarray,
            joint: int) -> None:
    """Give the joint's boxes that stay minus to a face-neighbouring joint of the two volumes."""
    left = kept.copy()
    universe_joints = theirs[theirs != joint]
    for _ in range(owner.shape[0]):
        if not left.any():
            return
        progressed = False
        for target in universe_joints:
            taking = left & near(owner == target, state.wrap, faces_only=True)
            if taking.any():
                owner[taking] = target
                marks[taking] |= 2
                left &= ~taking
                progressed = True
        if not progressed:
            break
    others = (owner >= 0) & (owner != joint)
    for place in np.argwhere(left):
        for radius in range(1, owner.shape[0]):
            window = tuple(slice(max(index - radius, 0), index + radius + 1) for index in place)
            found = owner[window][others[window]]
            if found.size:
                owner[tuple(place)] = int(found[0])
                marks[tuple(place)] |= 2
                break


def _join_twins(state: State, universe: int, volume: int, owner: np.ndarray, marks: np.ndarray) -> None:
    """Where the merged volume now has two joints with one neighbour, make them one joint."""
    joints = state.joints_of(universe, volume)
    sides = state.between[universe, joints]
    other = np.where(sides[:, 0] == volume, sides[:, 1], sides[:, 0])
    for neighbour in np.unique(other):
        twins = np.sort(joints[other == neighbour])
        if twins.size < 2 or neighbour < 0:
            continue
        first = int(twins[0])
        for twin in twins[1:]:
            moving = owner == twin
            owner[moving] = first
            marks[moving] |= 2
            state.size[universe, first] += state.size[universe, twin]
            state.speed[universe, first] += state.speed[universe, twin]
            state.kind[universe, twin] = 0
            state.size[universe, twin] = 0.0
            state.speed[universe, twin] = 0.0
            state.between[universe, twin] = -1

