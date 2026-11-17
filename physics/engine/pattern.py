"""Pattern: is the grid still a perfect A B A B pattern? Counts every way it could be broken.

Contract: `pattern(state)` reads the host's relations and returns a dict of ints, all 0
(and web_pieces 1) while the pattern holds (axiom 3):
  volumes, joints   living energy volumes and joints
  touching          energy volumes with a neighbour that is not a living joint (plus
                    touching plus, or a hole)
  unwrapped         energy volumes with one of the 26 directions holding no joint
  one_sided         neighbour entries not answered: a lists b but b does not list a
  dead_links        neighbour entries of living cells that point at an id that is no cell
  web_pieces        connected pieces of the joints through joint-to-joint neighbours
                    (1: one web)
  loose_ways        volumes on a joint's sides that are not living energy volumes
                    beside the joint, plus volumes whose state.ways_h is not the number
                    of ways they are an end of
  neighbours        the mean number of neighbours of an energy volume (26 on the lattice)
Pristine cells are checked from the lattice in one go, changed cells one by one.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from physics.engine.state import CENTRE, ENERGY_VOLUME, JOINT, State


def pattern(state: State) -> dict:
    """Count the breaks of the A B A B pattern in the host's relations."""
    kind = state.kind_h[:state.count]
    rows, cols, codes = _entries(state)
    alive = kind[rows] != 0
    rows, cols, codes = rows[alive], cols[alive], codes[alive]
    from_volume = kind[rows] == ENERGY_VOLUME
    bad = from_volume & (kind[cols] != JOINT)
    filled = np.zeros((state.count, 27), dtype=bool)
    filled[rows[from_volume], codes[from_volume]] = True
    volumes = np.flatnonzero(kind == ENERGY_VOLUME)
    keys = set((rows.astype(np.int64) * state.count + cols).tolist())
    back = cols.astype(np.int64) * state.count + rows
    one_sided = sum(1 for key in back.tolist() if key not in keys) if state.adj_over else 0
    both = (kind[rows] == JOINT) & (kind[cols] == JOINT)
    joints = np.flatnonzero(kind == JOINT)
    web = coo_matrix((np.ones(int(both.sum()), dtype=np.int8), (rows[both], cols[both])),
                     shape=(state.count, state.count))
    _, labels = connected_components(web, directed=False)
    seen = len(set(zip(rows[from_volume].tolist(), cols[from_volume].tolist())))
    return {"volumes": len(volumes), "joints": len(joints), "touching": len(np.unique(rows[bad])),
            "unwrapped": int((filled[volumes].sum(axis=1) < 26).sum()), "one_sided": int(one_sided),
            "dead_links": int((kind[cols] == 0).sum()), "web_pieces": len(np.unique(labels[joints])),
            "loose_ways": _loose_ways(state), "neighbours": seen / max(len(volumes), 1)}


def _entries(state: State) -> tuple:
    """Every neighbour entry (cell, neighbour, code): pristine cells from the lattice, changed ones from their lists."""
    side = state.side
    pristine = np.array([cell for cell in range(state.lattice) if cell not in state.adj_over], dtype=np.int64) \
        if state.adj_over else np.arange(state.lattice, dtype=np.int64)
    place = np.stack([pristine // (side * side), (pristine // side) % side, pristine % side], axis=1)
    all_codes = np.array([code for code in range(27) if code != CENTRE])
    steps = np.stack([all_codes // 9 - 1, (all_codes // 3) % 3 - 1, all_codes % 3 - 1], axis=1)
    there = (place[:, None, :] + steps[None, :, :]) % side
    rows = [np.repeat(pristine, 26)]
    cols = [((there[..., 0] * side + there[..., 1]) * side + there[..., 2]).reshape(-1)]
    codes = [np.tile(all_codes, len(pristine))]
    for cell, (ids, own_codes) in state.adj_over.items():
        rows.append(np.full(len(ids), cell, dtype=np.int64))
        cols.append(ids.astype(np.int64))
        codes.append(own_codes.astype(np.int64))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(codes)


def _loose_ways(state: State) -> int:
    """Volumes on a changed joint's sides that are not living volumes beside it, and wrong per-volume way counts."""
    loose = 0
    changed = {cell for cell in state.adj_over if state.kind_h[cell] == ENERGY_VOLUME}
    for joint, (first, second) in state.sides_over.items():
        beside = set(state.neighbours(joint)[0].tolist()) if first or second else set()
        loose += sum(1 for end in first + second if state.kind_h[end] != ENERGY_VOLUME or end not in beside)
        changed |= set(first) | set(second)
    for volume in changed:
        if state.kind_h[volume] != ENERGY_VOLUME:
            continue
        ends = 0
        for joint in state.joints_of(volume):
            first, second = state.sides(joint)
            ends += (volume in first) * (len(second) - (volume in second))
            ends += (volume in second) * (len(first) - (volume in first))
        loose += 0 if ends == state.ways_h[volume] else 1
    return loose
