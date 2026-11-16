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
plane (measured the short way round the edge from the plane when the universe wraps;
before 2026-11-17 the joint's layer just below the volume counted as past the plane
there) becoming a joint of the new half (marks bit 2, no sign change; its boxes in the
plane that touch both halves join the tear's joint), so two neighbouring volumes still
share exactly one joint; each piece of a shared joint keeps the squeeze the whole had:
its size is the size its own quanta want times the whole's size over the size the whole's
quanta wanted (a piece holds less energy, so by length = constant / energy it wants more
room; sharing size by boxes squeezed every piece past the collapse limit at once). The
tear's joint is born at the size its quanta want (cell_laws.py), not squeezed: owner,
2026-11-16, a fresh joint from a tear holds the least energy it will ever have and gains
as the volumes press on it (its push starts at settings.push, its swing at rest; joint
sizes are their own swing, never box counts, joint_swing.py; its boxes follow). ABAB holds after: the two halves' plus boxes lie two boxes
apart across the tear. A cut that leaves a half with fewer than least_cell_boxes boxes
is not made (a cell must hold a whirl).
state.splits and state.torn count splits and the quanta turned minus. Only the room the
volume and its joints lie in is read to the host and written back (cells_span.py,
window_read.py; the whole universe when that room reaches round it). Returns the
number of splits.
"""
from __future__ import annotations

import numpy as np

from physics.engine.cells_span import cells_span
from physics.engine.near import near
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.window_read import window_read
from physics.engine.window_write import window_write


def volume_split(state: State, settings: Settings) -> int:
    """Split, per universe, the biggest energy volume past split_across in two with a tear joint between."""
    if settings.split_across <= 0.0:
        return 0
    limit = settings.split_across ** 3
    big = (state.kind == 1) & (state.cell_boxes >= limit)
    if not big.any():
        return 0
    done = 0
    for universe in np.flatnonzero(big.any(axis=1)):
        volume = int(np.argmax(np.where(big[universe], state.cell_boxes[universe], -1)))
        done += _split_one(state, int(universe), volume, max(int(settings.least_cell_boxes), 1))
    return done


def _split_one(state: State, universe: int, volume: int, least: int) -> int:
    """Tear one volume, reading and writing only the room it and its joints lie in; returns 1 when cut."""
    room = cells_span(state, universe, [volume, *state.joints_of(universe, volume)])
    low, extent = (np.zeros(3, dtype=np.int64), np.full(3, state.side)) if room is None else room
    quanta, owner, marks = window_read(state, universe, low, extent)
    cut = _tear(state, universe, volume, quanta, owner, marks, least, state.wrap and room is None)
    if cut:
        window_write(state, universe, low, quanta, owner, marks)
    return cut


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


def _tear(state: State, universe: int, volume: int, quanta: np.ndarray, owner: np.ndarray, marks: np.ndarray,
          least: int, wrap: bool) -> int:
    """Cut one volume in two with a tear joint between, in the room the arrays hold (`wrap`: it is the whole universe, wrapping); returns 1 when cut."""
    places = np.argwhere(owner == volume)
    measured, shift = _along(places, owner.shape[0], wrap)
    spans = measured.max(axis=0) - measured.min(axis=0)
    axis = int(np.argmax(spans))
    cut = int(np.median(measured[:, axis]))
    low = measured[:, axis] < cut
    high = measured[:, axis] > cut
    if min(int(low.sum()), int(high.sum())) < least:
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
                     axis, cut, shift[axis], wrap)
    state.size[universe, tear] = _wants(state, universe, int(np.abs(quanta[owner == tear].astype(np.int64)).sum()))
    state.speed[universe, tear] = 0.0
    state.cell_clock[universe, [half, tear]] = state.cell_clock[universe, volume]
    state.splits[universe] += 1
    state.torn[universe] += torn
    return 1


def _share_joint(state: State, universe: int, joint: int, volume: int, half: int, tear: int, quanta: np.ndarray,
                 owner: np.ndarray,
                 marks: np.ndarray, lower_boxes: np.ndarray, upper_boxes: np.ndarray, axis: int, cut: int,
                 shift: int, wrap: bool) -> None:
    """Give a joint of the torn volume to the half it touches, or share it out along the plane when it touches both."""
    boxes = owner == joint
    by_lower = near(lower_boxes, wrap, faces_only=True)
    by_upper = near(upper_boxes, wrap, faces_only=True)
    touches_lower = bool((boxes & by_lower).any())
    touches_upper = bool((boxes & by_upper).any())
    sides = state.between[universe, joint]
    other = int(sides[1] if sides[0] == volume else sides[0])
    if touches_upper and not touches_lower:
        state.between[universe, joint] = _pair(other, half)
        return
    if not touches_upper:
        return
    past = np.indices(owner.shape)[axis] - shift - cut
    if wrap:
        past = (past + owner.shape[axis] // 2) % owner.shape[axis] - owner.shape[axis] // 2
    both = boxes & by_lower & by_upper & (past == 0)
    beyond = boxes & ~both & ((past > 0) | ((past == 0) & by_upper))
    twin = state.new_cell(universe, -1)
    state.between[universe, twin] = _pair(other, half)
    owner[beyond] = twin
    marks[beyond] |= 2
    owner[both] = tear
    marks[both] |= 2
    held = np.abs(quanta.astype(np.int64))
    whole = float(state.size[universe, joint])
    stretch = whole / _wants(state, universe, int(held[boxes].sum()))
    state.size[universe, twin] = stretch * _wants(state, universe, int(held[beyond].sum()))
    state.size[universe, joint] = stretch * _wants(state, universe, int(held[boxes & ~beyond & ~both].sum()))
    state.cell_clock[universe, twin] = state.cell_clock[universe, joint]


def _wants(state: State, universe: int, amount: int) -> float:
    """The size a joint holding `amount` quanta wants (cell_laws.py: (size_energy / quanta) ** 3)."""
    return (float(state.size_energy[universe, 1]) / max(amount, 1)) ** 3


def _pair(first: int, second: int) -> list:
    """Two cell ids, real ones first in rising order, -1 last."""
    return sorted((first, second), key=lambda cell: (cell < 0, cell))
