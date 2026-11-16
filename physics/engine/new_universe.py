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
batch holds that many universes, universe u built from seed + u. Only the uneven field
and the per-cell tables are made on the host; every per-box step runs on the GPU
(_start.py; 2026-11-17, before which a start of 18 million boxes took 128 s and tens of
bytes of host memory a box), with the same operations in the same order, so a seed
gives the same universe as before, to the last quantum. Raises ValueError for cells_per_side < 3, k < 2, a negative margin, box_quanta
< 1, a negative joint_unevenness and the uneven field's errors (filled side under 8).
"""
from __future__ import annotations

import itertools

import numpy as np
import warp as wp

from physics.engine._start import cell_ids, face_partner, fill_cells, held_boxes
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
            origin: float, filled: int, wrap: int, far: int, owner: wp.array3d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    here = wp.vec3(float(ti), float(tj), float(tk))
    best = float(1.0e30)
    chosen = int(-1)
    width = 2 * far + 1
    for index in range(width * width * width):
        offset = wp.vec3i(index // (width * width) - far, index // width % width - far, index % width - far)
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
    lattice = {"count": count, "period": period, "filled": filled, "wrap": wrap, "low": (side - filled) // 2,
               "origin": (0.0 if wrap else 1.0) + 0.5 * (boxes_per_cell_side - 1), "extent": cells_per_side + 2,
               "cells_per_side": cells_per_side, "boxes": boxes_per_cell_side}
    stacked = (universes * side, side, side)
    quanta = wp.zeros(stacked, dtype=wp.int32, device=device)
    owner = wp.full(stacked, -1, dtype=wp.int32, device=device)
    graphs = []
    for number in range(universes):
        field = uneven_field(filled, seed + number, slope, unevenness)
        np.exp(field, out=field)
        mean = field.mean()
        field *= box_quanta
        field /= mean
        energy = _block_energy(field, count, period, boxes_per_cell_side, wrap)
        field_d = wp.array(field, dtype=wp.float64, device=device)
        del field
        wanted = (count * boxes_per_cell_side) ** 3 * energy ** -3.0 / (energy ** -3.0).sum()
        graphs.append(_one_universe(lattice, wanted, energy, field_d, box_quanta, joint_unevenness, (seed + number, 1),
                                    number * side, owner, quanta))
        del field_d
    state = State.from_device(quanta, owner, graphs, seed=seed, wrap=wrap)
    cell_count(state)
    cell_laws(state, Settings())
    state.size = np.where(state.kind == -1, state.wanted, state.size)
    return state


def _one_universe(lattice: dict, wanted: np.ndarray, energy: np.ndarray, field, box_quanta: int,
                  joint_unevenness: float, rng_seed, first: int, owner, quanta) -> dict:
    """Cut, carve and fill one universe's boxes on the GPU; returns its cell graph (kind, between, place, size_energy)."""
    device = owner.device
    count, filled, wrap = lattice["count"], lattice["filled"], int(lattice["wrap"])
    raw, partner = _diagram(wanted, lattice, device)
    outer = wp.zeros(count ** 3, dtype=wp.int32, device=device)
    wp.launch(face_partner, dim=(filled,) * 3, device=device,
              inputs=[raw, count, wp.float64(lattice["period"]), wp.float64(lattice["origin"]), filled, wrap, partner,
                      outer])
    graph, lattice_ids, outer_id = _tables(lattice, outer.numpy() > 0)
    cells = graph["kind"].size
    boxes = wp.zeros(cells, dtype=wp.int32, device=device)
    spans = wp.array(np.tile(np.array([filled, -1, filled, -1], dtype=np.int32), (cells, 3)), dtype=wp.int32,
                     device=device)
    wp.launch(cell_ids, dim=(filled,) * 3, device=device,
              inputs=[raw, partner, count, lattice["extent"], lattice["cells_per_side"], filled, wrap,
                      wp.array(outer_id, dtype=wp.int32, device=device), first, lattice["low"], owner, boxes, spans])
    del raw, partner
    held = boxes.numpy().astype(np.float64)
    _joints_from(graph, lattice, held)
    energy = _settled(energy, held[lattice_ids].astype(np.int64))
    totals = np.zeros(cells)
    totals[lattice_ids] = energy
    joints = np.flatnonzero(graph["kind"] == -1)
    totals[joints] = _joint_amounts(held[joints], box_quanta, joint_unevenness, rng_seed)
    wp.launch(fill_cells, dim=cells, device=device,
              inputs=[wp.array(graph["kind"].astype(np.int32), dtype=wp.int32, device=device),
                      wp.array(totals, dtype=wp.float64, device=device), spans, filled, wrap, first, lattice["low"],
                      owner, field, quanta])
    graph["size"] = held
    lasting = joints[totals[joints] > 0]
    graph["size_energy"] = np.array([
        float(np.cbrt(held[lattice_ids].sum() / (energy ** -3.0).sum())),
        float(np.cbrt(held[lasting].sum() / (totals[lasting] ** -3.0).sum()))])
    return graph


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


def _diagram(wanted: np.ndarray, lattice: dict, device) -> tuple:
    """The power diagram's volume per box and each box's carve partner (-2 a plus box), weights fitted to the wanted sizes.

    A volume two lattice places off can only take a box from one a place nearer when its
    weight is more than 2 * period**2 higher, so while the weights lie closer together
    than that the 27 nearest places are searched, not 125: the same diagram.
    """
    count, period, filled = lattice["count"], lattice["period"], lattice["filled"]
    axis = np.arange(count) * period + lattice["origin"]
    centres = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    weights = np.zeros(len(wanted))
    centre_array = wp.array(centres, dtype=wp.vec3, device=device)
    raw = wp.zeros((filled,) * 3, dtype=wp.int32, device=device)
    partner = wp.zeros((filled,) * 3, dtype=wp.int32, device=device)
    counted = wp.zeros(len(wanted), dtype=wp.int32, device=device)
    for _ in range(_ROUNDS):
        near_enough = count >= 6 and float(np.ptp(weights.astype(np.float32))) < 2.0 * period * period - 1.0
        wp.launch(_assign, dim=(filled,) * 3, device=device,
                  inputs=[centre_array, wp.array(weights.astype(np.float32), dtype=wp.float32, device=device), count,
                          float(period), float(lattice["origin"]), filled, int(lattice["wrap"]),
                          1 if near_enough else 2, raw])
        wp.launch(_carve, dim=(filled,) * 3, device=device, inputs=[raw, filled, int(lattice["wrap"]), partner])
        counted.zero_()
        wp.launch(held_boxes, dim=(filled,) * 3, device=device, inputs=[raw, partner, counted])
        held = counted.numpy().astype(np.float64)
        if np.all(np.abs(held - wanted) <= _CLOSE * wanted):
            break
        weights += 0.4 * (np.cbrt(wanted) ** 2 - np.cbrt(np.maximum(held, 1.0)) ** 2)
    return raw, partner


def _tables(lattice: dict, has_outer: np.ndarray) -> tuple:
    """The cell table before its joints are known: volumes at their lattice ids, outer joints after all lattice ids.

    Returns (graph, each volume's lattice id, each volume's outer joint id or -1).
    """
    count, extent = lattice["count"], lattice["extent"]
    grid = np.array(list(itertools.product(range(count), repeat=3)))
    lattice_ids = ((2 + 2 * grid[:, 0]) * extent + 2 + 2 * grid[:, 1]) * extent + 2 + 2 * grid[:, 2]
    with_outer = np.flatnonzero(has_outer)
    cells = extent ** 3 + with_outer.size
    outer_id = np.full(count ** 3, -1, dtype=np.int32)
    outer_id[with_outer] = extent ** 3 + np.arange(with_outer.size)
    kind = np.zeros(cells, dtype=np.int8)
    between = np.full((cells, 2), -1, dtype=np.int32)
    places = np.full((cells, 3), -1, dtype=np.int32)
    kind[lattice_ids] = 1
    places[lattice_ids] = 2 + 2 * grid
    kind[outer_id[with_outer]] = -1
    between[outer_id[with_outer], 0] = lattice_ids[with_outer]
    return {"kind": kind, "between": between, "place": places}, lattice_ids, outer_id


def _joints_from(graph: dict, lattice: dict, held: np.ndarray) -> None:
    """Enter every joint that got boxes at a lattice place between two volumes: its kind, its two volumes, its place."""
    extent, cells_per_side = lattice["extent"], lattice["cells_per_side"]
    lattice_count = extent ** 3
    cells = np.flatnonzero((held[:lattice_count] > 0) & (graph["kind"][:lattice_count] == 0))
    place = np.stack(np.unravel_index(cells, (extent,) * 3), axis=1)
    odd = place % 2 == 1
    ends = []
    for way in (-1, 1):
        end = place + way * odd
        if lattice["wrap"]:
            end = (end - 1) % cells_per_side + 1
        ends.append((end[:, 0] * extent + end[:, 1]) * extent + end[:, 2])
    graph["kind"][cells] = -1
    graph["between"][cells] = np.sort(np.stack(ends, axis=1), axis=1)
    graph["place"][cells] = place


def _joint_amounts(held: np.ndarray, box_quanta: int, joint_unevenness: float, rng_seed) -> np.ndarray:
    """Each joint's whole quanta: the energy whose wanted size is its box count, box_quanta per box on average."""
    amounts = box_quanta * held.sum() * held ** (-1.0 / 3.0) / (held ** (-1.0 / 3.0)).sum()
    if joint_unevenness > 0.0:
        factor = np.exp(np.random.default_rng(rng_seed).normal(0.0, joint_unevenness, held.size))
        amounts = amounts * factor / factor.mean()
    return np.maximum(np.rint(amounts), held)
