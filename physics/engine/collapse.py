"""Collapse: a joint packed past its limit gives way, and its two energy volumes become one cell at once.

Contract: `collapse(state, settings, pressed=None)` collapses, most packed first, every
joint that lies between two energy volumes and whose energy concentration, minus quanta
per box (state.cell_quanta / state.cell_boxes, cell_count.py), is at least
collapse_packing, or whose own size (state.size, its swing's) is below its quanta over
collapse_packing (the swing's own test; such a joint does not swing, and before
2026-11-17 it never collapsed either while its boxes, a wall at least one box thick, kept
it under the limit: 943 of 4,416 joints of the fitted universe stood so at step 200), or
that `pressed` (bool per universe and cell id, joint_swing.py: its
swing reached the collapse packing within the step) marks. Owner, 2026-11-16: a joint
collapses only when its energy concentration gets too high, never because it is low on
energy or smaller than its energy wants (a fresh joint from a tear holds the least
energy it will ever have); when a joint collapses its two volumes become one cell at once. Joint J
between volumes A and B (A the lower id, it stays): every box of J turns plus, its minus
quanta becoming plus quanta of A, the same amount (bible rule 4: joint collapse is the
one place where pushing energy becomes pulling energy), marks bit 1; except a box of J
that has another volume's plus box or empty space or a wall among its 26 neighbours, or
a joint other than A's and B's on a face, or that bridges two parts of the joint web
(turning it would cut a piece of the web off): that box keeps its minus quanta and goes
to a neighbouring joint of A or B, else the nearest joint (marks bit 2), so the merged
volume stays wrapped on all 26 sides and the joints stay one web. Every box of B joins A
(bit 2): B and J are no more. A joint B had with a volume X becomes A's; where A and B
both had a joint with X the two become one joint (the higher id's boxes join the
lower's, bit 2, speeds added), so two neighbouring volumes still share exactly
one joint; the joined joint holds both joints' quanta, so it wants less room than the two
did (length = constant / energy), and its size is that wanted size times the two joints'
pooled stretch (their summed size over their summed wanted size): it is as squeezed as
they were together and pushes on as before, the mirror of a tear, whose pieces keep the
whole's squeeze (volume_split.py). Before 2026-11-17 the sizes were added instead: the
joined joint sat 16 times larger than its energy wants and pushed nothing (median squeeze
-0.99 after the fitted universe's collapses), and a volume that tore and fell back
together got its joints back 16 times or more bigger each time (ten steps running took
the joints' summed size from 1.9e4 to 1.4e42 boxes in a universe of 290,000). J's speed goes to the joints it shared springs with (speed_handover.py: the
springs only trade size), and so does the speed of a joint the joining left past its
limit, which swings no more and collapses when the next step opens. A's clock is the box weighted mean of A's and B's. tallies[u, 1] counts the
boxes turned plus, tallies[u, 3] their quanta, state.collapses the joints collapsed. The
boxes are changed on the host: only the room the joint, its two volumes and their joints
lie in is read and written back (cells_span.py), when the joint boxes left beside the
turning ones stay joined inside that room (then no piece of the web can be cut off) and
every box that stays minus finds a joint on its faces; otherwise the whole universe is
read, as before 2026-11-17 for every collapse. Returns the number of collapses.
"""
from __future__ import annotations

import numpy as np

from physics.engine.cells_span import cells_span
from physics.engine.group_label import group_label
from physics.engine.near import near
from physics.engine.settings import Settings
from physics.engine.speed_handover import speed_handover
from physics.engine.state import State
from physics.engine.window_read import window_read
from physics.engine.window_write import window_write


def collapse(state: State, settings: Settings, pressed: np.ndarray | None = None) -> int:
    """Collapse every joint packed past the limit (or pressed there by its swing); its two volumes become one."""
    packing = state.cell_quanta / np.maximum(state.cell_boxes, 1)
    over = packing >= settings.collapse_packing
    over |= (state.cell_quanta > 0) & (state.size < state.cell_quanta / settings.collapse_packing)
    if pressed is not None:
        over |= pressed
    if not over.any():
        return 0
    ready = (state.kind == -1) & over & (state.between.min(axis=-1) >= 0)
    if not ready.any():
        return 0
    tallies = state.tallies.numpy()
    done = 0
    for universe in np.flatnonzero(ready.any(axis=1)):
        for joint in sorted(np.flatnonzero(ready[universe]), key=lambda cell: -packing[universe, cell]):
            if state.kind[universe, joint] != -1 or state.between[universe, joint].min() < 0:
                continue
            speed_handover(state, settings, int(universe), [int(joint)])
            flipped, amount = _collapse_one(state, int(universe), int(joint))
            tallies[universe, 1] += flipped
            tallies[universe, 3] += amount
            state.collapses[universe] += 1
            done += 1
        joined = (state.kind[universe] == -1) & (state.cell_quanta[universe] > 0)
        joined &= state.size[universe] < state.cell_quanta[universe] / settings.collapse_packing
        speed_handover(state, settings, int(universe), np.flatnonzero(joined))
    state.tallies.assign(tallies)
    return done


def _collapse_one(state: State, universe: int, joint: int) -> tuple:
    """Collapse one joint in the room it, its volumes and their joints lie in; in the whole universe when that cannot be shown to give the same."""
    keep, gone = (int(cell) for cell in sorted(state.between[universe, joint]))
    theirs = np.concatenate([state.joints_of(universe, keep), state.joints_of(universe, gone)])
    room = cells_span(state, universe, [keep, gone, *theirs])
    if room is not None:
        quanta, owner, marks = window_read(state, universe, *room)
        result = _fall_in(state, universe, joint, quanta, owner, marks, False, True)
        if result is not None:
            window_write(state, universe, room[0], quanta, owner, marks)
            return result
    low, extent = np.zeros(3, dtype=np.int64), np.full(3, state.side)
    quanta, owner, marks = window_read(state, universe, low, extent)
    result = _fall_in(state, universe, joint, quanta, owner, marks, state.wrap, False)
    window_write(state, universe, low, quanta, owner, marks)
    return result


def _fall_in(state: State, universe: int, joint: int, quanta: np.ndarray, owner: np.ndarray, marks: np.ndarray,
             wrap: bool, room_only: bool):
    """Turn joint `joint` into its lower volume, B into A; returns (boxes turned plus, quanta turned plus).

    With `room_only` the arrays hold only a room round the cells; returns None, nothing
    changed but the arrays, when the room cannot show what the whole universe would give.
    """
    keep, gone = (int(cell) for cell in sorted(state.between[universe, joint]))
    theirs = np.concatenate([state.joints_of(universe, keep), state.joints_of(universe, gone)])
    boxes = owner == joint
    plus = quanta > 0
    foreign = plus & (owner != keep) & (owner != gone)
    blocked = near(foreign, wrap) | near(quanta == 0, wrap, wall=True)
    blocked |= near((quanta < 0) & ~np.isin(owner, theirs), wrap, faces_only=True)
    if room_only:
        turned = boxes & ~blocked
        if not _web_holds(quanta < 0, turned):
            return None
    else:
        turned = _keep_web(quanta < 0, boxes & ~blocked, wrap)
    if not _rehome(state, universe, owner, marks, boxes & ~turned, theirs, joint, wrap, room_only):
        return None
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


def _web_holds(minus: np.ndarray, turned: np.ndarray) -> bool:
    """True when the joint boxes left beside the turning ones stay joined inside the room.

    Every path through the web that crossed a turning box entered and left it by such a
    neighbour, so when those stay joined to each other nothing is cut off anywhere and
    _keep_web would hand back `turned` unchanged; False sends the collapse to the whole
    universe, where _keep_web decides.
    """
    left = minus & ~turned
    beside = left & near(turned, False, faces_only=True)
    labels, _ = group_label(left, False)
    return np.unique(labels[beside]).size <= 1


def _rehome(state: State, universe: int, owner: np.ndarray, marks: np.ndarray, kept: np.ndarray,
            theirs: np.ndarray, joint: int, wrap: bool, room_only: bool) -> bool:
    """Give the joint's boxes that stay minus to a face-neighbouring joint of the two volumes, else the nearest joint.

    Returns False, with `room_only`, when a box would need the nearest joint looked for
    (a search the room cannot stand in for); True otherwise.
    """
    left = kept.copy()
    universe_joints = theirs[theirs != joint]
    for _ in range(owner.shape[0]):
        if not left.any():
            return True
        progressed = False
        for target in universe_joints:
            taking = left & near(owner == target, wrap, faces_only=True)
            if taking.any():
                owner[taking] = target
                marks[taking] |= 2
                left &= ~taking
                progressed = True
        if not progressed:
            break
    if not left.any():
        return True
    if room_only:
        return False
    others = (owner >= 0) & (owner != joint) & np.isin(owner, np.flatnonzero(state.kind[universe] == -1))
    for place in np.argwhere(left):
        for radius in range(1, owner.shape[0]):
            window = tuple(slice(max(index - radius, 0), index + radius + 1) for index in place)
            found = owner[window][others[window]]
            if found.size:
                owner[tuple(place)] = int(found[0])
                marks[tuple(place)] |= 2
                break
    return True


def _keep_web(minus: np.ndarray, turned: np.ndarray, wrap: bool) -> np.ndarray:
    """The joint boxes that may turn plus without cutting a piece off the joint web; the rest stay minus.

    Joint boxes that bridge two parts of the web stay minus: a piece of the web that turning
    would cut off takes back the turning boxes on its faces, layer by layer, until it is
    joined again.
    """
    before, _ = group_label(minus, wrap)
    for _ in range(minus.shape[0]):
        after, _ = group_label(minus & ~turned, wrap)
        orphans = np.zeros(minus.shape, dtype=bool)
        for piece in np.unique(before[turned & (before > 0)]):
            inside = after[(before == piece) & (after > 0)]
            if inside.size == 0:
                continue
            parts, sizes = np.unique(inside, return_counts=True)
            if parts.size > 1:
                orphans |= np.isin(after, parts[np.arange(parts.size) != np.argmax(sizes)])
        if not orphans.any():
            return turned
        back = turned & near(orphans, wrap, faces_only=True)
        if not back.any():
            return turned
        turned = turned & ~back
    return turned


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
            _pool(state, universe, first, int(twin))
            state.speed[universe, first] += state.speed[universe, twin]
            state.kind[universe, twin] = 0
            state.size[universe, twin] = 0.0
            state.speed[universe, twin] = 0.0
            state.between[universe, twin] = -1



def _pool(state: State, universe: int, first: int, twin: int) -> None:
    """Give joint `first` the twin's quanta and the size they want together, times the two joints' pooled stretch."""
    sizes = state.size[universe, [first, twin]]
    wanted = state.wanted[universe, [first, twin]]
    amount = int(state.cell_quanta[universe, [first, twin]].sum())
    if amount <= 0 or wanted.sum() <= 0.0:
        state.size[universe, first] = float(sizes.sum())
        return
    together = (float(state.size_energy[universe, 1]) / amount) ** 3
    stretch = float(sizes.sum()) / float(wanted.sum())
    state.size[universe, first] = stretch * together
    state.wanted[universe, first] = together
    state.cell_quanta[universe, first] = amount
    state.cell_quanta[universe, twin] = 0
    state.wanted[universe, twin] = 0.0
