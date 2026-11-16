"""New universe: energy volumes wrapped in a thin web of joints, each volume as big as its energy wants (one correction).

Contract: `new_universe(cells_per_side=16, boxes_per_cell_side=10, seed=1, unevenness=0.3,
slope=-3, margin=0, box_quanta=1000, universes=1, device="cuda", joint_unevenness=0.0)`
returns a State (state.py), phase 0. The start lattice has cells_per_side places
per axis, numbered 1 .. n; a place is an energy volume when all three numbers are even
(and inside 2 .. n - 1 unless the universe wraps), a block boxes_per_cell_side boxes
across (owner, 2026-11-16: energy volumes start about 10 boxes across), the other
numbers are joint layers one box thick. With margin 0, an even cells_per_side and an
even side (cells_per_side / 2 * (boxes_per_cell_side + 1)) the universe wraps round at
its edges (opposite faces meet: no edge swells into nothing; the fluid's paired faces
need an even side);
otherwise `margin` empty boxes lie round the filled cube, the side rounded up to a
multiple of 8.
Energy: the uneven field g over the filled cube (uneven_field.py: seed, slope, rms
unevenness; ripples at every size) gives each box box_quanta * exp(g) / mean(exp(g));
an energy volume's energy is the sum over its block, in whole quanta. The one correction
(bible, "The foundation" 1: one correction lets sizes follow energy): the volumes'
constant of length = constant / energy is fixed so that their wanted sizes add up to the
blocks' room, and each volume is given that many boxes: the filled cube is cut round the
volumes' block centres by a power diagram (a box goes to the volume nearest it by
squared distance minus the volume's weight, round the edges when the universe wraps),
the weights adjusted round after round until every volume's plus boxes are its wanted
size within 2 percent (at most 200 rounds; the carving below keeps it near 6 percent).
Whole boxes cannot hit every size, so the correction then hands each volume the energy
whose wanted size is the boxes it got (the total kept; each volume's energy moves by the
few percent its box count missed), so every volume starts at the size its energy wants.
The mortar is carved out one box thick: a
box that has among its 26 neighbours a box of a lower-numbered volume, or empty space or
a wall, is a joint box. Its joint lies between its volume and that lower volume when the
two are face neighbours on the lattice, else between its volume and the lattice face
neighbour whose centre is nearest the box; a box with only empty space or a wall beside
it belongs to its volume's outer joint (between the volume and -1). So no two volumes
touch on a face, an edge or a corner, every volume is wrapped on all 26 sides, each
volume has its 6 lattice face neighbours as its neighbours (fewer on the outer layer)
and each joint lies between two volumes (the cell graph, state.py). A volume's id is
its lattice place's id (a * (n + 2) + b) * (n + 2) + c; a joint between two volumes has
the id of the lattice place between them, an outer joint an id after all lattice ids. A
volume's quanta are its energy, shared over its plus boxes in proportion to exp(g)
there (whole quanta, largest remainders, at least 1 a box). Each joint gets the energy
whose wanted size is its box count, with the joints' constant fixed so that they hold
box_quanta quanta per box on average (times exp(h) / mean(exp(h)) per joint, h normal
with rms joint_unevenness from default_rng((seed + u, 1)), when joint_unevenness > 0),
shared evenly over its boxes. So every cell starts at the size its energy wants; the
energy itself is not at rest: a volume's quanta follow the uneven field inside it, so
its fluid, which has its own pressure (fluid_potential.py), starts flowing at once and
presses its joints unevenly (wall_press.py; axiom 1, owner 2026-11-16: energy is always
moving). Raises ValueError when boxes_per_cell_side ** 3 is below
Settings().least_cell_boxes (a cell too small to hold a whirl). With universes > 1 the
batch holds that many universes, universe u built from seed + u. Raises ValueError for cells_per_side < 3, k < 2, a negative margin, box_quanta
< 1, a negative joint_unevenness and the uneven field's errors (filled side under 8).
"""
from __future__ import annotations

import itertools

import numpy as np
import warp as wp

from physics.engine.cell_count import cell_count
from physics.engine.cell_laws import cell_laws
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.uneven_field import uneven_field

_ROUNDS = 200
_CLOSE = 0.02
_NONE = 2147483647


@wp.kernel
def _assign(centres: wp.array(dtype=wp.vec3), weights: wp.array(dtype=wp.float32), count: int, period: float,
            origin: float, filled: int, wrap: int, owner: wp.array3d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    here = wp.vec3(float(ti), float(tj), float(tk))
    best = float(1.0e30)
    chosen = int(-1)
    for index in range(125):
        offset = wp.vec3i(index // 25 - 2, index // 5 % 5 - 2, index % 5 - 2)
        grid = wp.vec3i(0, 0, 0)
        valid = True
        for axis in range(3):
            near = int(wp.round((here[axis] - origin) / period)) + offset[axis]
            if wrap == 1:
                near = ((near % count) + count) % count
            elif near < 0 or near >= count:
                valid = False
            grid[axis] = near
        if valid:
            cell = (grid[0] * count + grid[1]) * count + grid[2]
            apart = here - centres[cell]
            if wrap == 1:
                for axis in range(3):
                    apart[axis] = apart[axis] - float(filled) * wp.round(apart[axis] / float(filled))
            power = wp.dot(apart, apart) - weights[cell]
            if power < best:
                best = power
                chosen = cell
    owner[ti, tj, tk] = chosen


@wp.kernel
def _carve(raw: wp.array3d(dtype=wp.int32), filled: int, wrap: int, partner: wp.array3d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    own = raw[ti, tj, tk]
    lowest_face = int(_NONE)
    lowest = int(_NONE)
    outside = int(0)
    for index in range(27):
        if index != 13:
            offset = wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
            spot = wp.vec3i(ti + offset[0], tj + offset[1], tk + offset[2])
            valid = True
            for axis in range(3):
                if spot[axis] < 0 or spot[axis] >= filled:
                    if wrap == 1:
                        spot[axis] = (spot[axis] + filled) % filled
                    else:
                        valid = False
            if not valid:
                outside = 1
            else:
                other = raw[spot[0], spot[1], spot[2]]
                if other < own:
                    lowest = wp.min(lowest, other)
                    if wp.abs(offset[0]) + wp.abs(offset[1]) + wp.abs(offset[2]) == 1:
                        lowest_face = wp.min(lowest_face, other)
    if lowest_face < _NONE:
        partner[ti, tj, tk] = lowest_face
    elif lowest < _NONE:
        partner[ti, tj, tk] = lowest
    elif outside == 1:
        partner[ti, tj, tk] = -1
    else:
        partner[ti, tj, tk] = -2


def new_universe(cells_per_side: int = 16, boxes_per_cell_side: int = 10, seed: int = 1, unevenness: float = 0.3,
                 slope: float = -3.0, margin: int = 0, box_quanta: int = 1000, universes: int = 1,
                 device: str = "cuda", joint_unevenness: float = 0.0) -> State:
    """Seeded energy volumes sized by their energy in a thin joint web, every cell at its wanted size."""
    if cells_per_side < 3 or boxes_per_cell_side < 2:
        raise ValueError(f"need at least 3 cells per side of at least 2 boxes, got {cells_per_side}, "
                         f"{boxes_per_cell_side}")
    if not joint_unevenness >= 0.0:
        raise ValueError(f"joint_unevenness must be >= 0, got {joint_unevenness}")
    if boxes_per_cell_side ** 3 < Settings().least_cell_boxes:
        raise ValueError(f"a cell needs at least {Settings().least_cell_boxes} boxes to hold a whirl, got "
                         f"{boxes_per_cell_side} across ({boxes_per_cell_side ** 3} boxes)")
    if box_quanta < 1:
        raise ValueError(f"box_quanta must be at least 1, got {box_quanta}")
    margin = int(margin)
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    period = boxes_per_cell_side + 1
    wrap = margin == 0 and cells_per_side % 2 == 0 and (cells_per_side // 2 * period) % 2 == 0
    count = cells_per_side // 2 if wrap else (cells_per_side - 1) // 2
    filled = count * period + (0 if wrap else 1)
    side = filled if margin == 0 else -(-(filled + 2 * margin) // 8) * 8
    low = (side - filled) // 2
    origin = (0.0 if wrap else 1.0) + 0.5 * (boxes_per_cell_side - 1)
    extent = cells_per_side + 2
    grid = np.array(list(itertools.product(range(count), repeat=3)))
    lattice_ids = ((2 + 2 * grid[:, 0]) * extent + 2 + 2 * grid[:, 1]) * extent + 2 + 2 * grid[:, 2]
    quanta = np.zeros((universes, side, side, side), dtype=np.int32)
    graphs = []
    for number in range(universes):
        raw = np.exp(uneven_field(filled, seed + number, slope, unevenness))
        field = box_quanta * raw / raw.mean()
        energy = _block_energy(field, count, period, boxes_per_cell_side, wrap)
        wanted = (count * boxes_per_cell_side) ** 3 * energy ** -3.0 / (energy ** -3.0).sum()
        volume_of, partner = _diagram(wanted, count, period, origin, filled, wrap, device)
        partner = _face_partner(volume_of, partner, grid, count, period, origin, filled, wrap)
        energy = _settled(energy, np.bincount(volume_of[partner == -2], minlength=energy.size))
        graph, owner = _graph(volume_of, partner, lattice_ids, grid, count, extent, cells_per_side, wrap)
        inner = (slice(low, low + filled),) * 3
        quanta[number][inner] = _fill(owner, graph["kind"], field, volume_of, partner, energy, box_quanta,
                                      joint_unevenness, (seed + number, 1))
        full = np.full((side,) * 3, -1, dtype=np.int32)
        full[inner] = owner
        graph["owner"] = full
        held = np.bincount(volume_of[partner == -2], minlength=energy.size).astype(np.float64)
        graph["size_energy"] = np.array([float(np.cbrt(held.sum() / (energy ** -3.0).sum())),
                                         _joint_constant(quanta[number], full, graph["kind"])])
        graphs.append(graph)
    state = State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=seed, device=device,
                            lattice=graphs, wrap=wrap)
    cell_count(state)
    cell_laws(state, Settings())
    state.size = np.where(state.kind == -1, state.wanted, state.size)
    return state


def _block_energy(field: np.ndarray, count: int, period: int, boxes: int, wrap: bool) -> np.ndarray:
    """Each volume's energy: the field summed over its start block, whole quanta."""
    start = 0 if wrap else 1
    sums = np.zeros(count ** 3)
    for index, place in enumerate(itertools.product(range(count), repeat=3)):
        block = tuple(slice(start + value * period, start + value * period + boxes) for value in place)
        sums[index] = field[block].sum()
    return np.rint(sums)


def _settled(energy: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Each volume's whole quanta set to the energy whose wanted size is the boxes it holds, the total kept."""
    shaped = np.maximum(held, 1).astype(np.float64) ** (-1.0 / 3.0)
    exact = shaped * energy.sum() / shaped.sum()
    whole = np.floor(exact)
    rest = int(energy.sum() - whole.sum())
    whole[np.argsort(-(exact - whole))[:rest]] += 1.0
    return whole


def _diagram(wanted: np.ndarray, count: int, period: int, origin: float, filled: int, wrap: bool, device) -> tuple:
    """The power diagram's volume per box and each box's carve partner (-2 a plus box), weights fitted to the wanted sizes."""
    axis = np.arange(count) * period + origin
    centres = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    weights = np.zeros(len(wanted))
    centre_array = wp.array(centres, dtype=wp.vec3, device=device)
    raw = wp.zeros((filled,) * 3, dtype=wp.int32, device=device)
    partner = wp.zeros((filled,) * 3, dtype=wp.int32, device=device)
    for _ in range(_ROUNDS):
        wp.launch(_assign, dim=(filled,) * 3, device=device,
                  inputs=[centre_array, wp.array(weights.astype(np.float32), dtype=wp.float32, device=device), count,
                          float(period), float(origin), filled, int(wrap), raw])
        wp.launch(_carve, dim=(filled,) * 3, device=device, inputs=[raw, filled, int(wrap), partner])
        volume_of, sides = raw.numpy(), partner.numpy()
        held = np.bincount(volume_of[sides == -2], minlength=len(wanted)).astype(np.float64)
        if np.all(np.abs(held - wanted) <= _CLOSE * wanted):
            break
        weights += 0.4 * (np.cbrt(wanted) ** 2 - np.cbrt(np.maximum(held, 1.0)) ** 2)
    return volume_of, sides


def _face_partner(volume_of: np.ndarray, partner: np.ndarray, grid: np.ndarray, count: int, period: int,
                  origin: float, filled: int, wrap: bool) -> np.ndarray:
    """Each joint box's partner made a lattice face neighbour of its volume: the nearest one when the lowest volume beside it is not."""
    carved = partner >= 0
    own = volume_of[carved]
    lower = partner[carved]
    apart = grid[lower] - grid[own]
    if wrap and count > 2:
        apart = (apart + count // 2) % count - count // 2
    face = np.abs(apart).sum(axis=1) == 1
    places = np.argwhere(carved)[~face].astype(np.float64)
    own_far = own[~face]
    best = np.full(own_far.size, -1, dtype=np.int64)
    nearest = np.full(own_far.size, np.inf)
    for axis in range(3):
        for step in (-1, 1):
            neighbour = grid[own_far].copy()
            neighbour[:, axis] += step
            if wrap:
                neighbour %= count
            valid = (neighbour >= 0).all(axis=1) & (neighbour < count).all(axis=1)
            index = (neighbour[:, 0] * count + neighbour[:, 1]) * count + neighbour[:, 2]
            gap = places - (np.clip(neighbour, 0, count - 1) * period + origin)
            if wrap:
                gap -= filled * np.round(gap / filled)
            distance = np.where(valid, (gap ** 2).sum(axis=1), np.inf)
            closer = distance < nearest
            nearest[closer] = distance[closer]
            best[closer] = index[closer]
    lower = lower.copy()
    lower[~face] = best
    fixed = partner.copy()
    fixed[carved] = lower
    return fixed


def _graph(volume_of: np.ndarray, partner: np.ndarray, lattice_ids: np.ndarray, grid: np.ndarray, count: int,
           extent: int, cells_per_side: int, wrap: bool) -> tuple:
    """Kind, between and place per cell id, and the owner of every filled box."""
    plus = partner == -2
    pairs = np.stack([volume_of[~plus], partner[~plus]], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    lattice_count = extent ** 3
    joint_ids = np.zeros(len(unique), dtype=np.int64)
    extra = lattice_count
    for row, (one, two) in enumerate(unique):
        place = _between_place(grid, one, two, count, cells_per_side, wrap) if two >= 0 else None
        if place is None:
            joint_ids[row] = extra
            extra += 1
        else:
            joint_ids[row] = (place[0] * extent + place[1]) * extent + place[2]
    kind = np.zeros(extra, dtype=np.int8)
    between = np.full((extra, 2), -1, dtype=np.int32)
    places = np.full((extra, 3), -1, dtype=np.int32)
    kind[lattice_ids] = 1
    places[lattice_ids] = 2 + 2 * grid
    for row, (one, two) in enumerate(unique):
        cell = joint_ids[row]
        kind[cell] = -1
        ends = sorted([int(lattice_ids[one])] + ([int(lattice_ids[two])] if two >= 0 else []))
        between[cell, :len(ends)] = ends
        if cell < lattice_count:
            places[cell] = np.unravel_index(cell, (extent,) * 3)
    owner = np.where(plus, lattice_ids[volume_of], 0).astype(np.int32)
    owner[~plus] = joint_ids[inverse.ravel()]
    return {"kind": kind, "between": between, "place": places}, owner


def _between_place(grid: np.ndarray, one: int, two: int, count: int, cells_per_side: int, wrap: bool):
    """The lattice place halfway between two volumes that are face neighbours on the lattice, else None."""
    apart = grid[two] - grid[one]
    if wrap and count > 2:
        apart = (apart + count // 2) % count - count // 2
    if np.abs(apart).sum() != 1:
        return None
    place = 2 + 2 * grid[one] + apart
    if wrap:
        place = (place - 1) % cells_per_side + 1
    return place


def _fill(owner: np.ndarray, kind: np.ndarray, field: np.ndarray, volume_of: np.ndarray, partner: np.ndarray,
          energy: np.ndarray, box_quanta: int, joint_unevenness: float, rng_seed) -> np.ndarray:
    """Signed whole quanta per filled box: volumes their energy over their plus boxes, joints the energy their size wants."""
    cube = np.zeros(owner.shape, dtype=np.int64)
    plus = partner == -2
    volume_whole = _share(volume_of[plus], field[plus], energy)
    cube[plus] = volume_whole
    joints = np.flatnonzero(kind == -1)
    held = np.bincount(owner[~plus], minlength=kind.size)[joints].astype(np.float64)
    amounts = box_quanta * held.sum() * held ** (-1.0 / 3.0) / (held ** (-1.0 / 3.0)).sum()
    if joint_unevenness > 0.0:
        factor = np.exp(np.random.default_rng(rng_seed).normal(0.0, joint_unevenness, joints.size))
        amounts = amounts * factor / factor.mean()
    per_joint = np.zeros(kind.size)
    per_joint[joints] = np.maximum(np.rint(amounts), held)
    ids = owner[~plus]
    cube[~plus] = -_share(ids, np.ones(ids.size), per_joint)
    return cube.astype(np.int32)


def _share(groups: np.ndarray, weights: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Whole numbers per member, each group's total shared in proportion to the weights, largest remainders, at least 1."""
    sums = np.bincount(groups, weights=weights, minlength=totals.size)
    exact = weights / sums[groups] * totals[groups]
    whole = np.maximum(np.floor(exact), 1.0)
    rest = np.rint(totals - np.bincount(groups, weights=whole, minlength=totals.size)).astype(np.int64)
    order = np.lexsort((-(exact - np.floor(exact)), groups))
    starts = np.searchsorted(groups[order], np.arange(totals.size))
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size) - starts[groups[order]]
    return (whole + (rank < rest[groups])).astype(np.int64)


def _joint_constant(cube: np.ndarray, owner: np.ndarray, kind: np.ndarray) -> float:
    """The joints' constant C of length = C / energy with which their wanted sizes add up to the boxes they hold."""
    held = owner >= 0
    amounts = np.bincount(owner[held], weights=np.abs(cube[held]).astype(np.float64), minlength=kind.size)
    boxes = np.bincount(owner[held], minlength=kind.size).astype(np.float64)
    joints = (kind == -1) & (amounts > 0)
    return float(np.cbrt(boxes[joints].sum() / (amounts[joints] ** -3.0).sum()))
