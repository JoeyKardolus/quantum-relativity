"""Move corners: the forced cell lattice's corners move so each cell's size follows its energy.

Contract: `move_corners(state, settings)` does, for a state with a cell lattice
(state.has_lattice; without one it returns at once):
  1. counts every cell's whole quanta ignoring sign into state.cell_quanta (int64
     atomics, exact) and writes its volume in boxes into state.cell_size
  2. gives every real cell a pressure from how much smaller or larger it is than its
     energy wants (bible rule 4, length = constant / energy): a cell of q quanta wants
     to be size_energy / q boxes across, so its wanted size is (size_energy / q)**3
     boxes and its ratio r = wanted size / size (size: the cell's volume in boxes from
     its moving corners, kuhn_tet.py). An energy volume presses with volume_stiffness
     volume_rest r**2 / 2 (its fluid's push at volume_rest when r = 1), a joint with
     push joint_rest s / (s + 1) r**(s + 1), s = push_steepness (the steep push); a cell
     with no quanta and the ring of empty space press with 0. An energy volume is also
     drawn by the pull, -quanta * grad(pull) at its centre (the pull wave's field,
     central differences of the boxes round the centre), shared over its eight corners
  3. moves every real corner by corner_mobility * force * step_time, force = the sum
     over its eight cells of pressure * d(cell size)/d(corner) plus the pull's shares,
     at most max_corner_step boxes, kept inside the universe (0 .. side); the ring's
     corners follow the outer real corners, flat against the walls
  4. keeps every real cell right side out and every joint at least a box thick: a
     cell whose tets (kuhn_tet.py) would fall below MIN_TET boxes of volume, or a
     joint that would get thinner than MIN_JOINT_ACROSS (1 box) across an axis on
     which its lattice index is odd (the axes it lies between volumes on; each of its
     four edges along it, measured across its low face) and thinner than it was, puts
     its corners back where they were, repeated until no cell is bad (at most CHECKS
     rounds, then no corner moves this step). Owner, 2026-11-16: between two energy
     volumes there is always a joint box, on faces, edges and corners; a joint thinner
     than a box left its boxes lagging behind its corners
So a cell's size is set by its own energy pressing against its neighbours' (no set
size, no minimum): more energy makes a cell want less room, so it presses less and its
neighbours take room from it, and it ends smaller and denser (owner, 2026-11-16: "more
energy, smaller cell"); a joint pressed smaller than its energy wants pushes back
steeply and swells; the pull draws volumes together and so squeezes the joints between
them.
Pressures and forces are float64: a joint far smaller than its energy wants (a start of
small cells) presses past float32's range, and an overflowed force froze every corner.
Every force is gathered per corner from its eight cells in a fixed order, no float
atomics, so a universe moves the same alone or side by side in a batch. Boxes follow
their cell only through box_targets.py, drain.py and hand_over.py. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.cell_kind import cell_kind
from physics.engine.cell_place import cell_id, cell_place
from physics.engine.inside import inside
from physics.engine.kuhn_corner import kuhn_corner
from physics.engine.kuhn_tet import kuhn_tet
from physics.engine.settings import Settings
from physics.engine.state import State

MIN_TET = 1.0e-3
MIN_JOINT_ACROSS = 1.0
CHECKS = 8

wp.set_module_options({"max_unroll": 1})


@wp.func
def _tet_volume(start: wp.vec3, one: wp.vec3, two: wp.vec3, end: wp.vec3) -> float:
    return wp.dot(one - start, wp.cross(two - start, end - start)) / 6.0


@wp.kernel
def _count(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), side: int,
           cell_quanta: wp.array2d(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    held = quanta[ti, tj, tk]
    if held != 0:
        wp.atomic_add(cell_quanta, ti // side, owner[ti, tj, tk], wp.int64(wp.abs(held)))


@wp.func
def _slope(pull: wp.array3d(dtype=wp.float32), home: int, side: int, box: wp.vec3i, axis: int) -> float:
    """Central difference of the pull along `axis` at `box`, 0 where a neighbour is off the grid."""
    up = box
    down = box
    up[axis] = up[axis] + 1
    down[axis] = down[axis] - 1
    if not inside(up, home, side) or not inside(down, home, side):
        return 0.0
    return 0.5 * (pull[up[0], up[1], up[2]] - pull[down[0], down[1], down[2]])


@wp.kernel
def _press(corners: wp.array3d(dtype=wp.vec3), cell_quanta: wp.array2d(dtype=wp.int64),
           pull: wp.array3d(dtype=wp.float32), side: int, extent: wp.vec3i, counts: wp.array(dtype=wp.vec3i),
           size_energy: float,
           volume_rest: float, volume_stiffness: float, joint_rest: float, push: float, push_steepness: float,
           pressure: wp.array2d(dtype=wp.float64), drawn: wp.array2d(dtype=wp.vec3),
           cell_size: wp.array2d(dtype=wp.float32)):
    home, cell = wp.tid()
    kind = cell_kind(cell, extent, counts[home])
    pressure[home, cell] = wp.float64(0.0)
    cell_size[home, cell] = 0.0
    drawn[home, cell] = wp.vec3(0.0, 0.0, 0.0)
    if kind == 0:
        return
    place = cell_place(cell, extent)
    size = float(0.0)
    centre = wp.vec3(0.0, 0.0, 0.0)
    for tet in range(6):
        start, one, two, end = kuhn_tet(corners, home, place, tet, extent)
        size += _tet_volume(start, one, two, end)
        if tet == 0:
            centre = 0.5 * (start + end)
    cell_size[home, cell] = size
    amount = float(cell_quanta[home, cell])
    if size <= 0.0 or amount <= 0.0:
        return
    wanted_side = wp.float64(size_energy) / wp.float64(amount)
    ratio = wanted_side * wanted_side * wanted_side / wp.float64(size)
    if kind > 0:
        pressure[home, cell] = wp.float64(volume_stiffness * volume_rest) * ratio * ratio / wp.float64(2.0)
        box = wp.vec3i(home * side + wp.clamp(int(wp.floor(centre[0])), 0, side - 1),
                       wp.clamp(int(wp.floor(centre[1])), 0, side - 1),
                       wp.clamp(int(wp.floor(centre[2])), 0, side - 1))
        drawn[home, cell] = -amount * wp.vec3(_slope(pull, home, side, box, 0), _slope(pull, home, side, box, 1),
                                              _slope(pull, home, side, box, 2))
    else:
        power = push_steepness + 1.0
        pressure[home, cell] = wp.float64(push * joint_rest * push_steepness / power) * wp.pow(ratio, wp.float64(power))


@wp.kernel
def _propose(corners: wp.array3d(dtype=wp.vec3), pressure: wp.array2d(dtype=wp.float64),
             drawn: wp.array2d(dtype=wp.vec3), side: int, extent: wp.vec3i, counts: wp.array(dtype=wp.vec3i),
             scale: float, max_step: float, proposal: wp.array3d(dtype=wp.vec3)):
    ti, tj, tk = wp.tid()
    rows = extent[0] + 1
    home = ti // rows
    vertex = wp.vec3i(ti - home * rows, tj, tk)
    old = corners[ti, tj, tk]
    proposal[ti, tj, tk] = old
    cells = counts[home]
    for axis in range(3):
        if vertex[axis] == 0 or vertex[axis] >= cells[axis] + 2:
            return
    force = wp.vec3d(0.0, 0.0, 0.0)
    for around in range(8):
        bits = wp.vec3i(around // 4, around // 2 % 2, around % 2)
        place = vertex - bits
        cell = cell_id(place, extent)
        squeeze = pressure[home, cell]
        force += wp.vec3d(drawn[home, cell] / 8.0)
        if squeeze != wp.float64(0.0):
            for tet in range(6):
                start, one, two, end = kuhn_tet(corners, home, place, tet, extent)
                for slot in range(4):
                    if kuhn_corner(tet, slot) == bits:
                        grad = wp.vec3(0.0, 0.0, 0.0)
                        if slot == 1:
                            grad = wp.cross(two - start, end - start) / 6.0
                        elif slot == 2:
                            grad = wp.cross(end - start, one - start) / 6.0
                        elif slot == 3:
                            grad = wp.cross(one - start, two - start) / 6.0
                        else:
                            grad = -(wp.cross(two - start, end - start) + wp.cross(end - start, one - start)
                                     + wp.cross(one - start, two - start)) / 6.0
                        force += squeeze * wp.vec3d(grad)
    push_by = wp.float64(scale) * force
    length = wp.length(push_by)
    if length > wp.float64(max_step):
        push_by = push_by * (wp.float64(max_step) / length)
    moved = old + wp.vec3(push_by)
    top = float(side)
    proposal[ti, tj, tk] = wp.vec3(wp.clamp(moved[0], 0.0, top), wp.clamp(moved[1], 0.0, top),
                                   wp.clamp(moved[2], 0.0, top))


@wp.kernel
def _ring(proposal: wp.array3d(dtype=wp.vec3), side: int, extent: wp.vec3i, counts: wp.array(dtype=wp.vec3i)):
    ti, tj, tk = wp.tid()
    rows = extent[0] + 1
    home = ti // rows
    vertex = wp.vec3i(ti - home * rows, tj, tk)
    cells = counts[home]
    on_ring = int(0)
    nearest = vertex
    for axis in range(3):
        last = cells[axis] + 2
        if vertex[axis] == 0 or vertex[axis] >= last:
            on_ring = 1
        nearest[axis] = wp.clamp(vertex[axis], 1, last - 1)
    if on_ring == 0:
        return
    position = proposal[home * rows + nearest[0], nearest[1], nearest[2]]
    for axis in range(3):
        if vertex[axis] == 0:
            position[axis] = 0.0
        elif vertex[axis] >= cells[axis] + 2:
            position[axis] = float(side)
    proposal[ti, tj, tk] = position


@wp.func
def _across(corners: wp.array3d(dtype=wp.vec3), home: int, place: wp.vec3i, axis: int, extent: wp.vec3i) -> float:
    """The cell's least thickness along `axis`: its four edges along it, each measured across its low face."""
    rows = extent[0] + 1
    first = (axis + 1) % 3
    second = (axis + 2) % 3
    low = wp.vec3i(home * rows + place[0], place[1], place[2])
    step = wp.vec3i(0, 0, 0)
    step[axis] = 1
    one = wp.vec3i(0, 0, 0)
    one[first] = 1
    two = wp.vec3i(0, 0, 0)
    two[second] = 1
    both = one + two
    normal = wp.cross(corners[low[0] + both[0], low[1] + both[1], low[2] + both[2]] - corners[low[0], low[1], low[2]],
                      corners[low[0] + two[0], low[1] + two[1], low[2] + two[2]]
                      - corners[low[0] + one[0], low[1] + one[1], low[2] + one[2]])
    if wp.length(normal) <= 0.0:
        return 0.0
    normal = wp.normalize(normal)
    thinnest = float(1.0e30)
    for edge in range(4):
        at = low + one * (edge // 2) + two * (edge % 2)
        top = at + step
        span = corners[top[0], top[1], top[2]] - corners[at[0], at[1], at[2]]
        thinnest = wp.min(thinnest, wp.abs(wp.dot(span, normal)))
    return thinnest


@wp.func
def _thinned(corners: wp.array3d(dtype=wp.vec3), proposal: wp.array3d(dtype=wp.vec3), home: int, place: wp.vec3i,
             extent: wp.vec3i) -> bool:
    """True when the move leaves the joint thinner than MIN_JOINT_ACROSS along an axis it lies across, and thinner than it was."""
    for axis in range(3):
        if place[axis] % 2 == 1:
            after = _across(proposal, home, place, axis, extent)
            if after < MIN_JOINT_ACROSS and after < _across(corners, home, place, axis, extent):
                return True
    return False


@wp.kernel
def _check(corners: wp.array3d(dtype=wp.vec3), proposal: wp.array3d(dtype=wp.vec3), extent: wp.vec3i,
           counts: wp.array(dtype=wp.vec3i), flags: wp.array3d(dtype=wp.int32), bad: wp.array(dtype=wp.int32)):
    home, cell = wp.tid()
    kind = cell_kind(cell, extent, counts[home])
    if kind == 0:
        return
    place = cell_place(cell, extent)
    smallest = float(1.0e30)
    for tet in range(6):
        start, one, two, end = kuhn_tet(proposal, home, place, tet, extent)
        smallest = wp.min(smallest, _tet_volume(start, one, two, end))
    if smallest > MIN_TET and not (kind == -1 and _thinned(corners, proposal, home, place, extent)):
        return
    rows = extent[0] + 1
    wp.atomic_add(bad, 0, 1)
    for around in range(8):
        flags[home * rows + place[0] + around // 4, place[1] + around // 2 % 2, place[2] + around % 2] = 1


@wp.kernel
def _revert(corners: wp.array3d(dtype=wp.vec3), proposal: wp.array3d(dtype=wp.vec3),
            flags: wp.array3d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    if flags[ti, tj, tk] == 1:
        proposal[ti, tj, tk] = corners[ti, tj, tk]
        flags[ti, tj, tk] = 0


def move_corners(state: State, settings: Settings) -> None:
    """Count each cell's quanta and move the lattice corners by the cells' pressures and the pull."""
    if not state.has_lattice:
        return
    device = state.quanta.device
    extent = wp.vec3i(*state.extent)
    count = state.cell_quanta.shape[1]
    state.cell_quanta.zero_()
    wp.launch(_count, dim=state.shape, device=device, inputs=[state.quanta, state.owner, state.side, state.cell_quanta])
    pressure = wp.zeros((state.universes, count), dtype=wp.float64, device=device)
    drawn = wp.zeros((state.universes, count), dtype=wp.vec3, device=device)
    wp.launch(_press, dim=(state.universes, count), device=device,
              inputs=[state.corners, state.cell_quanta, state.pull, state.side, extent, state.counts,
                      settings.size_energy, settings.volume_rest, settings.volume_stiffness, settings.joint_rest, settings.push,
                      settings.push_steepness, pressure, drawn, state.cell_size])
    proposal = wp.empty_like(state.corners)
    wp.launch(_propose, dim=state.corners.shape, device=device,
              inputs=[state.corners, pressure, drawn, state.side, extent, state.counts,
                      settings.corner_mobility * settings.step_time, settings.max_corner_step, proposal])
    flags = wp.zeros(state.corners.shape, dtype=wp.int32, device=device)
    bad = wp.zeros(1, dtype=wp.int32, device=device)
    for _ in range(CHECKS):
        wp.launch(_ring, dim=state.corners.shape, device=device, inputs=[proposal, state.side, extent, state.counts])
        bad.zero_()
        wp.launch(_check, dim=(state.universes, count), device=device,
                  inputs=[state.corners, proposal, extent, state.counts, flags, bad])
        if int(bad.numpy()[0]) == 0:
            wp.copy(state.corners, proposal)
            return
        wp.launch(_revert, dim=state.corners.shape, device=device, inputs=[state.corners, proposal, flags])
