"""Start grid: the perfect A B A B lattice as tables of relations, built once.

Contract: `start_grid(side, seed, spare, arrays, host, device)` returns the `Grid` of a
wrapping lattice of `side ** 3` cells: an energy volume where all three lattice indices
are even, a joint everywhere else (axiom 3: plus never touches plus, minus wraps every
plus and forms one web). It writes `kind`, `made_kind`, 26 neighbour entries a cell and
every way (1 through a face joint, 2 through an edge joint, 4 through a corner joint,
all in pass 0), worked with numpy for the whole lattice at once. Room:
`capacity = count * (1 + spare)` cells and the same share of spare way slots. The
lattice indices are used here and nowhere after: from now on a cell has only its
relations (axiom 3: no geometry of its own). It writes no energy and no size.
The start's ids are laid out `(x * side + y) * side + z`; `start_energy` hands the
uneven field to the volumes in rising id order, which is that order.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.direction_codes import CENTRE
from physics.engine.axiom_3_structure.definitions.grid import Grid
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_3_structure.logic.lay_lattice import lay_lattice

_CODES = np.array([code for code in range(27) if code != CENTRE], dtype=np.int32)


def _lattice_kinds(side: int) -> np.ndarray:
    """The kind of every start cell: a volume where all three indices are even, else a joint."""
    along = np.arange(side) % 2 == 0
    volume = along[:, None, None] & along[None, :, None] & along[None, None, :]
    return np.where(volume, ENERGY_VOLUME, JOINT).astype(np.int32).ravel()


def _lattice_neighbours(side: int) -> np.ndarray:
    """The 26 neighbour ids of every start cell, one column per code in rising order, wrapping round."""
    along = np.arange(side)
    first, second, third = np.meshgrid(along, along, along, indexing="ij")
    ids = np.empty((side ** 3, len(_CODES)), dtype=np.int32)
    for column, code in enumerate(_CODES.tolist()):
        there_first = (first + code // 9 - 1) % side
        there_second = (second + (code // 3) % 3 - 1) % side
        there_third = (third + code % 3 - 1) % side
        ids[:, column] = ((there_first * side + there_second) * side + there_third).ravel()
    return ids


def _lattice_ways(kind: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Every way `(a, b, mouth_a, mouth_b, joint)`: two volumes on opposite sides of one joint, a the lower id."""
    joints = np.flatnonzero(kind == JOINT).astype(np.int32)
    rows = []
    for column, code in enumerate(_CODES.tolist()):
        if code > CENTRE:
            continue
        one = ids[joints, column]
        other = ids[joints, len(_CODES) - 1 - column]
        real = (kind[one] == ENERGY_VOLUME) & (kind[other] == ENERGY_VOLUME) & (one != other)
        one, other, through = one[real], other[real], joints[real]
        lower = one < other
        rows.append(np.stack([np.where(lower, one, other), np.where(lower, other, one),
                              np.where(lower, 26 - code, code), np.where(lower, code, 26 - code), through], axis=1))
    return np.concatenate(rows).astype(np.int32)


def start_grid(side: int, seed: int, spare: float, arrays: dict, host: dict, device: str) -> Grid:
    """The Grid holding the start lattice: kinds, 26 neighbour entries a cell and every way."""
    if spare < 0.0:
        raise ValueError(f"spare must be 0 or more, got {spare}")
    side = int(side)
    if side < 4 or side % 2:
        raise ValueError(f"side must be even and at least 4, got {side}")
    kind = _lattice_kinds(side)
    ids = _lattice_neighbours(side)
    ways = _lattice_ways(kind, ids)
    count = side ** 3
    grid = Grid(side, int(np.ceil(count * (1.0 + spare))), int(np.ceil(len(ways) * (1.0 + spare))), arrays, host,
                seed, device)
    kinds = np.zeros(grid.capacity, dtype=np.int32)
    kinds[:count] = kind
    grid.kind.assign(kinds)
    grid.made_kind[:count] = kind
    lay_lattice(grid, ids, _CODES, ways)
    return grid
