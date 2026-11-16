"""Volume split: an energy volume grown too big tears in two, and the tear becomes a new joint (new lattice planes).

Contract: `volume_split(state, settings)` does, for a state with a cell lattice and
settings.split_across > 0, in every universe on its own (none: returns at once):
  1. finds the energy volumes whose size across, the cube root of their volume in boxes
     from their corners (state.cell_size, measured this step), has reached
     split_across (bible rule 8: the trigger is size, not density; a split keeps
     density, so a density trigger would split forever) and takes the largest; it
     tears across its longest axis (the four corner edges along each axis, averaged):
     the axis along which its neighbours pull from the farthest apart
  2. inserts two lattice planes through the whole universe after that volume's index
     a on that axis (the forced lattice grows only by whole planes: neighbours are
     fixed by index): every cell at index a is cut in three, a (the low part), a + 1
     (a TEAR boxes thin middle) and a + 2 (the high part), cells past a move up by two
     indices, and cell_kind.py makes a + 1 a joint and a + 2 an energy volume where a
     is one. So the too-big volume, and every other volume at index a of that layer,
     becomes two volumes with one new joint between them; every joint at index a
     becomes three joints; ABAB holds by index (every volume wrapped by joints on all
     26 sides, two neighbouring volumes share exactly one joint, the joints one web).
     The new corners lie on the old edges, the middle plane's two TEAR apart round
     each edge's midpoint
  3. gives each box of a cut cell to the part that covers its centre (cell_score.py;
     a lagging box to the part it lies deepest in). Boxes keep their quanta; a plus
     box that falls in a volume's new middle joint is the tear: its plus quanta turn
     minus, the same amount (bible rule 8: "the tear becomes new joints"; the mirror
     of joint collapse, where minus turns plus; owner question, see the overlay). A
     high-part box with a box of its low twin among its 26 neighbours joins the tear
     too, until none touch, so a joint box always lies between the two halves
  4. keeps the split only when every cut cell's three parts hold quanta and no part
     turns inside out; else it puts everything back and the volume waits
Each half holds about half the quanta, so it wants about eight times the room (move_
corners.py: size = size_energy / quanta): a split makes new room (owner, 2026-11-16).
The amount ignoring sign is exact; the tear's quanta are counted in state.torn and the
splits in state.splits; changed boxes get bit 8 in marks. A universe's lattice grows
alone; the batch's storage extent grows when a universe outgrows it, so a universe
splits the same alone as side by side. Returns None.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.cell_kind import cell_kind
from physics.engine.cell_place import cell_id, cell_place
from physics.engine.cell_score import cell_score
from physics.engine.lattice_kinds import lattice_kinds
from physics.engine.settings import Settings
from physics.engine.state import State

TEAR = 1.25
MARGIN = 1.0
MIN_TET = 1.0e-3
_PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
_PARITY = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)


@wp.kernel
def _renumber(ids: wp.array3d(dtype=wp.int32), side: int, old_extent: wp.vec3i, new_extent: wp.vec3i,
              universe: int, axis: int, index: int):
    ti, tj, tk = wp.tid()
    place = cell_place(ids[ti, tj, tk], old_extent)
    if ti // side == universe and place[axis] > index:
        place[axis] = place[axis] + 2
    ids[ti, tj, tk] = cell_id(place, new_extent)


@wp.kernel
def _assign(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32),
            target: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8), corners: wp.array3d(dtype=wp.vec3),
            side: int, extent: wp.vec3i, cells: wp.vec3i, universe: int, axis: int, index: int,
            torn: wp.array(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    row = universe * side + ti
    own = owner[row, tj, tk]
    place = cell_place(own, extent)
    if place[axis] != index:
        return
    point = wp.vec3(float(ti) + 0.5, float(tj) + 0.5, float(tk) + 0.5)
    best = float(-1.0e31)
    chosen = own
    for part in range(3):
        cut = place
        cut[axis] = index + part
        cell = cell_id(cut, extent)
        score = cell_score(corners, universe, cell, extent, point)
        if score > best:
            best = score
            chosen = cell
    if chosen == own:
        return
    held = quanta[row, tj, tk]
    if held > 0 and cell_kind(chosen, extent, cells) == -1:
        quanta[row, tj, tk] = -held
        wp.atomic_add(torn, 0, wp.int64(held))
    owner[row, tj, tk] = chosen
    target[row, tj, tk] = chosen
    marks[row, tj, tk] = marks[row, tj, tk] | wp.uint8(8)


@wp.kernel
def _touching(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), side: int, extent: wp.vec3i,
              universe: int, axis: int, index: int, flags: wp.array3d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    row = universe * side + ti
    flags[ti, tj, tk] = 0
    if quanta[row, tj, tk] <= 0:
        return
    place = cell_place(owner[row, tj, tk], extent)
    if place[axis] != index + 2:
        return
    twin = place
    twin[axis] = index
    low = cell_id(twin, extent)
    for near in range(27):
        offset = wp.vec3i(near // 9 - 1, near // 3 % 3 - 1, near % 3 - 1)
        there = wp.vec3i(ti, tj, tk) + offset
        if near != 13 and there[0] >= 0 and there[1] >= 0 and there[2] >= 0 and there[0] < side \
                and there[1] < side and there[2] < side:
            if quanta[universe * side + there[0], there[1], there[2]] > 0 \
                    and owner[universe * side + there[0], there[1], there[2]] == low:
                flags[ti, tj, tk] = 1


@wp.kernel
def _tear(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32),
          target: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8), side: int, extent: wp.vec3i,
          universe: int, axis: int, flags: wp.array3d(dtype=wp.int32), torn: wp.array(dtype=wp.int64),
          changed: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    if flags[ti, tj, tk] == 0:
        return
    row = universe * side + ti
    place = cell_place(owner[row, tj, tk], extent)
    place[axis] = place[axis] - 1
    middle = cell_id(place, extent)
    held = quanta[row, tj, tk]
    quanta[row, tj, tk] = -held
    owner[row, tj, tk] = middle
    target[row, tj, tk] = middle
    marks[row, tj, tk] = marks[row, tj, tk] | wp.uint8(8)
    wp.atomic_add(torn, 0, wp.int64(held))
    wp.atomic_add(changed, 0, 1)


def volume_split(state: State, settings: Settings) -> None:
    """Tear the largest too-big energy volume of every universe in two, the tear a new joint."""
    if not state.has_lattice or settings.split_across <= 0.0:
        return
    for universe in range(state.universes):
        found = _too_big(state, settings, universe)
        if found is not None:
            _split(state, universe, *found)


def _own_corners(state: State, universe: int) -> np.ndarray:
    """One universe's lattice vertices, its own (cells + 3) per axis, as a host array."""
    vertices = tuple(count + 1 for count in state.extent)
    corners = state.corners.numpy().reshape((state.universes,) + vertices + (3,))[universe]
    counts = state.cells[universe]
    return corners[:counts[0] + 3, :counts[1] + 3, :counts[2] + 3].copy()


def _too_big(state: State, settings: Settings, universe: int):
    """(lattice place, axis) of the universe's largest volume at or past split_across, or None."""
    counts = tuple(int(count) for count in state.cells[universe])
    own = tuple(count + 2 for count in counts)
    sizes = state.cell_size.numpy()[universe].reshape(state.extent)[:own[0], :own[1], :own[2]]
    across = np.cbrt(np.maximum(sizes, 0.0).astype(np.float64))
    across[lattice_kinds(counts) != 1] = 0.0
    if across.max() < settings.split_across:
        return None
    place = np.unravel_index(int(np.argmax(across)), own)
    corners = _own_corners(state, universe)
    lengths = [_edge_lengths(corners, axis)[place].mean() for axis in range(3)]
    return tuple(int(value) for value in place), int(np.argmax(lengths))


def _edge_lengths(corners: np.ndarray, axis: int) -> np.ndarray:
    """Per cell, the mean length of its four corner edges along `axis`."""
    edges = np.linalg.norm(np.diff(corners, axis=axis), axis=-1)
    others = [other for other in range(3) if other != axis]
    for other in others:
        edges = 0.5 * (np.take(edges, range(edges.shape[other] - 1), axis=other)
                       + np.take(edges, range(1, edges.shape[other]), axis=other))
    return edges


def _cut_corners(corners: np.ndarray, axis: int, index: int):
    """The universe's vertices with two planes inserted inside cell layer `index`, or None if an edge is too short."""
    low = np.take(corners, index, axis=axis)
    high = np.take(corners, index + 1, axis=axis)
    span = high - low
    length = np.linalg.norm(span, axis=-1, keepdims=True)
    if length.min() < TEAR + 2.0 * MARGIN:
        return None
    half = 0.5 * TEAR / length
    first = low + span * (0.5 - half)
    second = low + span * (0.5 + half)
    planes = np.stack([first, second], axis=axis).astype(np.float32)
    return np.concatenate([np.take(corners, range(index + 1), axis=axis), planes,
                           np.take(corners, range(index + 1, corners.shape[axis]), axis=axis)], axis=axis)


def _right_side_out(corners: np.ndarray, axis: int, index: int) -> bool:
    """True when every Kuhn tet of the three cut layers' real cells keeps more than MIN_TET volume."""
    real = [slice(1, size - 1) for size in corners.shape[:3]]
    real[axis] = slice(index, index + 4)
    layer = corners[tuple(real)]
    for permutation, parity in zip(_PERMUTATIONS, _PARITY):
        path = [np.zeros(3, dtype=int)]
        for step_axis in permutation:
            path.append(path[-1] + np.eye(3, dtype=int)[step_axis])
        points = [layer[bits[0]:layer.shape[0] - 1 + bits[0], bits[1]:layer.shape[1] - 1 + bits[1],
                        bits[2]:layer.shape[2] - 1 + bits[2]] for bits in path]
        volume = np.einsum("...i,...i->...", points[1] - points[0],
                           np.cross(points[2] - points[0], points[3] - points[0])) / 6.0
        if (parity * volume).min() <= MIN_TET:
            return False
    return True


def _split(state: State, universe: int, place: tuple, axis: int) -> None:
    """Insert two lattice planes after `place` on `axis` in one universe and hand the cut cells' boxes to their parts."""
    index = place[axis]
    corners = _cut_corners(_own_corners(state, universe), axis, index)
    if corners is None or not _right_side_out(corners, axis, index):
        return
    kept = _keep(state)
    counts = state.cells.copy()
    counts[universe, axis] += 2
    old_extent = state.extent
    _grow(state, counts, universe, corners)
    device = state.quanta.device
    for ids in (state.owner, state.target):
        wp.launch(_renumber, dim=state.shape, device=device,
                  inputs=[ids, state.side, wp.vec3i(*old_extent), wp.vec3i(*state.extent), universe, axis, index])
    extent = wp.vec3i(*state.extent)
    cells = wp.vec3i(*(int(count) for count in counts[universe]))
    torn = wp.zeros(1, dtype=wp.int64, device=device)
    cube = (state.side,) * 3
    wp.launch(_assign, dim=cube, device=device,
              inputs=[state.quanta, state.owner, state.target, state.marks, state.corners, state.side, extent, cells,
                      universe, axis, index, torn])
    flags = wp.zeros(cube, dtype=wp.int32, device=device)
    changed = wp.zeros(1, dtype=wp.int32, device=device)
    while True:
        changed.zero_()
        wp.launch(_touching, dim=cube, device=device,
                  inputs=[state.quanta, state.owner, state.side, extent, universe, axis, index, flags])
        wp.launch(_tear, dim=cube, device=device,
                  inputs=[state.quanta, state.owner, state.target, state.marks, state.side, extent, universe, axis,
                          flags, torn, changed])
        if int(changed.numpy()[0]) == 0:
            break
    if not _parts_hold_quanta(state, universe, axis, index):
        _restore(state, kept)
        return
    state.splits[universe] += 1
    state.torn[universe] += int(torn.numpy()[0])


def _parts_hold_quanta(state: State, universe: int, axis: int, index: int) -> bool:
    """True when every real cell of the three cut layers holds at least one box with quanta."""
    rows = state.universe(universe)
    quanta = state.quanta[rows.start:rows.stop].numpy()
    owner = state.owner[rows.start:rows.stop].numpy()[quanta != 0]
    held = np.bincount(owner.ravel(), minlength=int(np.prod(state.extent))).reshape(state.extent)
    counts = state.cells[universe]
    kinds = np.zeros(state.extent, dtype=np.int8)
    kinds[:counts[0] + 2, :counts[1] + 2, :counts[2] + 2] = lattice_kinds(counts)
    layers = [slice(None)] * 3
    layers[axis] = slice(index, index + 3)
    real = kinds[tuple(layers)] != 0
    return bool(np.all(held[tuple(layers)][real] > 0))


def _grow(state: State, counts: np.ndarray, universe: int, corners: np.ndarray) -> None:
    """Set the universe's new cell counts and corners, growing the batch's storage extent when it must."""
    extent = tuple(int(value) + 2 for value in counts.max(axis=0))
    vertices = tuple(count + 1 for count in extent)
    old_vertices = tuple(count + 1 for count in state.extent)
    every = state.corners.numpy().reshape((state.universes,) + old_vertices + (3,))
    grown = np.empty((state.universes,) + vertices + (3,), dtype=np.float32)
    for home in range(state.universes):
        own = corners if home == universe else every[home][:counts[home][0] + 3, :counts[home][1] + 3,
                                                           :counts[home][2] + 3]
        pad = [(0, vertices[axis] - own.shape[axis]) for axis in range(3)] + [(0, 0)]
        grown[home] = np.pad(own, pad, mode="edge")
    device = state.quanta.device
    state.corners = wp.array(grown.reshape((-1,) + vertices[1:] + (3,)), dtype=wp.vec3, device=device)
    state.cells = counts
    state.counts = wp.array(counts.astype(np.int32), dtype=wp.vec3i, device=device)
    state.extent = extent
    cell_count = extent[0] * extent[1] * extent[2]
    state.cell_quanta = wp.zeros((state.universes, cell_count), dtype=wp.int64, device=device)
    state.cell_size = wp.zeros((state.universes, cell_count), dtype=wp.float32, device=device)


def _keep(state: State) -> dict:
    """Copies of everything a split changes, to put back when it fails."""
    return {"quanta": wp.clone(state.quanta), "owner": wp.clone(state.owner), "target": wp.clone(state.target),
            "marks": wp.clone(state.marks), "corners": state.corners, "cells": state.cells, "counts": state.counts,
            "extent": state.extent, "cell_quanta": state.cell_quanta, "cell_size": state.cell_size}


def _restore(state: State, kept: dict) -> None:
    """Put back what _keep copied."""
    for name in ("quanta", "owner", "target", "marks"):
        wp.copy(getattr(state, name), kept[name])
    for name in ("corners", "cells", "counts", "extent", "cell_quanta", "cell_size"):
        setattr(state, name, kept[name])
