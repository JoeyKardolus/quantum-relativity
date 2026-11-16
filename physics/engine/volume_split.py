"""Volume split: an energy volume grown too big tears in two, and only that volume splits; the tear becomes the new joint.

Contract: `volume_split(state, settings)` splits, in each universe, the energy volume
with the most boxes once its size across (cube root of its box count) reaches
split_across (bible rule 8: the trigger is size; 0 switches splits off). The volume V is
cut by a plane one box thick across its longest extent, at the median of its boxes along
that axis (round the edge when the universe wraps): its boxes in the plane are the tear,
whose plus quanta turn minus, amount kept, and become a new joint between the two halves
(owner, 2026-11-16: a tear turns the middle of the torn volume's plus quanta into minus,
the mirror of collapse); the boxes past the plane become a new energy volume, the rest
stay V. Only V's boxes change cell or sign (marks bit 8); no other volume is touched.
Each joint V had keeps its boxes and lies between its other volume and the half it
touches; a joint touching both halves is shared out along the plane, its boxes past the
plane becoming a joint of the new half (marks bit 2, no sign change; its boxes in the
plane that touch both halves join the tear's joint), so two neighbouring volumes still
share exactly one joint; each piece of a shared joint keeps the squeeze the whole had:
its size is the size its own quanta want times the whole's size over the size the whole's
quanta wanted (a piece holds less energy, so by length = constant / energy it wants more
room; sharing size by boxes squeezed every piece past the collapse limit at once). The
tear's joint gets its plane's share of the whole's size by boxes (joint sizes are their
own swing, never box counts, joint_swing.py). ABAB holds after: the two halves' plus boxes lie two boxes
apart across the tear. A cut that leaves a half without boxes is not made.
state.splits and state.torn count splits and the quanta turned minus. Returns the
number of splits.
"""
from __future__ import annotations

import numpy as np

from physics.engine.near import near
from physics.engine.settings import Settings
from physics.engine.state import State


def volume_split(state: State, settings: Settings) -> int:
    """Split, per universe, the biggest energy volume past split_across in two with a tear joint between."""
    if settings.split_across <= 0.0:
        return 0
    limit = settings.split_across ** 3
    big = (state.kind == 1) & (state.cell_boxes >= limit)
    if not big.any():
        return 0
    quanta, owner, marks = state.quanta.numpy(), state.owner.numpy(), state.marks.numpy()
    done = 0
    for universe in np.flatnonzero(big.any(axis=1)):
        volume = int(np.argmax(np.where(big[universe], state.cell_boxes[universe], -1)))
        rows = state.universe(int(universe))
        done += _tear(state, int(universe), volume, quanta[rows], owner[rows], marks[rows])
    for name, values in (("quanta", quanta), ("owner", owner), ("marks", marks)):
        getattr(state, name).assign(values)
    return done


def _along(places: np.ndarray, side: int, wrap: bool) -> tuple:
    """Per axis the volume's coordinates measured from just past its widest gap (round the edge), and the shift."""
    if not wrap:
        return places, np.zeros(3, dtype=np.int64)
    shift = np.zeros(3, dtype=np.int64)
    for axis in range(3):
        used = np.zeros(side, dtype=bool)
        used[places[:, axis]] = True
        if used.all():
            continue
        free = np.flatnonzero(~used)
        doubled = np.concatenate([free, free + side])
        gaps = np.diff(doubled)
        widest = int(np.argmax(gaps))
        shift[axis] = (doubled[widest] + 1) % side
    return (places - shift) % side, shift


def _tear(state: State, universe: int, volume: int, quanta: np.ndarray, owner: np.ndarray, marks: np.ndarray) -> int:
    """Cut one volume in two with a tear joint between; returns 1 when cut."""
    side = state.side
    places = np.argwhere(owner == volume)
    measured, shift = _along(places, side, state.wrap)
    spans = measured.max(axis=0) - measured.min(axis=0)
    axis = int(np.argmax(spans))
    cut = int(np.median(measured[:, axis]))
    low = measured[:, axis] < cut
    high = measured[:, axis] > cut
    if not low.any() or not high.any():
        return 0
    half = state.new_cell(universe, 1)
    tear = state.new_cell(universe, -1)
    state.between[universe, tear] = sorted((volume, half))
    middle = places[measured[:, axis] == cut]
    upper = places[high]
    owner[tuple(upper.T)] = half
    marks[tuple(upper.T)] |= 8
    torn = int(quanta[tuple(middle.T)].astype(np.int64).sum())
    quanta[tuple(middle.T)] = -quanta[tuple(middle.T)]
    owner[tuple(middle.T)] = tear
    marks[tuple(middle.T)] |= 8
    lower_boxes = owner == volume
    upper_boxes = owner == half
    for joint in state.joints_of(universe, volume):
        if joint == tear:
            continue
        _share_joint(state, universe, int(joint), volume, half, tear, quanta, owner, marks, lower_boxes, upper_boxes,
                     axis, cut, shift[axis])
    state.size[universe, tear] = float(len(middle))
    state.cell_clock[universe, [half, tear]] = state.cell_clock[universe, volume]
    state.splits[universe] += 1
    state.torn[universe] += torn
    return 1


def _share_joint(state: State, universe: int, joint: int, volume: int, half: int, tear: int, quanta: np.ndarray,
                 owner: np.ndarray,
                 marks: np.ndarray, lower_boxes: np.ndarray, upper_boxes: np.ndarray, axis: int, cut: int,
                 shift: int) -> None:
    """Give a joint of the torn volume to the half it touches, or share it out along the plane when it touches both."""
    boxes = owner == joint
    by_lower = near(lower_boxes, state.wrap, faces_only=True)
    by_upper = near(upper_boxes, state.wrap, faces_only=True)
    touches_lower = bool((boxes & by_lower).any())
    touches_upper = bool((boxes & by_upper).any())
    sides = state.between[universe, joint]
    other = int(sides[1] if sides[0] == volume else sides[0])
    if touches_upper and not touches_lower:
        state.between[universe, joint] = _pair(other, half)
        return
    if not touches_upper:
        return
    coordinate = (np.indices(owner.shape)[axis] - shift) % state.side
    both = boxes & by_lower & by_upper & (coordinate == cut)
    beyond = boxes & ~both & ((coordinate > cut) | ((coordinate == cut) & by_upper))
    twin = state.new_cell(universe, -1)
    state.between[universe, twin] = _pair(other, half)
    owner[beyond] = twin
    marks[beyond] |= 2
    owner[both] = tear
    marks[both] |= 2
    held = np.abs(quanta.astype(np.int64))
    total = max(int(boxes.sum()), 1)
    whole = float(state.size[universe, joint])
    stretch = whole / _wants(state, universe, int(held[boxes].sum()))
    state.size[universe, twin] = stretch * _wants(state, universe, int(held[beyond].sum()))
    state.size[universe, tear] += whole * int(both.sum()) / total
    state.size[universe, joint] = stretch * _wants(state, universe, int(held[boxes & ~beyond & ~both].sum()))
    state.cell_clock[universe, twin] = state.cell_clock[universe, joint]


def _wants(state: State, universe: int, amount: int) -> float:
    """The size a joint holding `amount` quanta wants (cell_laws.py: (size_energy / quanta) ** 3)."""
    return (float(state.size_energy[universe, 1]) / max(amount, 1)) ** 3


def _pair(first: int, second: int) -> list:
    """Two cell ids, real ones first in rising order, -1 last."""
    return sorted((first, second), key=lambda cell: (cell < 0, cell))
