"""Placed: where every own cell is drawn and how wide, read off the relations and the sizes.

A cell has no place of its own, so the picture's places are worked out from what the engine
holds: each joint says how far apart its two volumes stand along its axis (the squeeze's
`apart`), and its part of each volume's side says how far off that side's middle it lies.
`settle` finds the volumes' middles that agree best with all the joints at once (they
cannot all agree: sizes change, space is not a lattice); a new volume starts where its joint
puts it. A volume is drawn as wide as its size, a joint between its two volumes.
The picture is drawn `span` times smaller once the cluster outgrows the window.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_5_packing.logic.squeeze import squeeze

ROUNDS = 40  # times the middles are moved toward what their joints say, a frame


def settle(store, spots: np.ndarray) -> tuple:
    """Every cell's middle (cells, 3) from the last ones `spots`, its size, and the joints (cell, before, after, axis)."""
    squeeze(store)
    held = store.cell_quanta.numpy()[:store.cells]
    size = QUANTA_A_BOX * store.across ** 3 / np.maximum(np.abs(held), 1)
    apart = np.zeros(store.cells)
    apart[store.own_cell.numpy()[:store.count]] = store.own_squeeze.numpy()[:store.count, 2]
    ends, parts = store.joint_ends.numpy()[:store.cells], store.joint_part.numpy()[:store.cells]
    joints = np.flatnonzero((ends[:, 2] >= 0) & (apart > 0))
    before, after, axis = ends[joints].T
    lies = np.arange(len(joints))
    aim = np.zeros((len(joints), 3))  # from the volume before to the one after
    aim[lies, axis] = apart[joints]
    for end, volume, sign in ((0, before, 1), (1, after, -1)):  # off the middles of their sides
        for at in (0, 1):
            off = parts[joints, end, 2 * at] + parts[joints, end, 2 * at + 1] / 2 - 0.5
            aim[lies, (axis + 1 + at) % 3] += sign * off * size[volume]
    middle, known = np.zeros((store.cells, 3)), np.zeros(store.cells, dtype=bool)
    middle[:len(spots)], known[:len(spots)] = spots, True
    known[0] = True
    for _ in range(len(joints)):  # a new volume starts where a joint to a known one puts it
        new = known[before] != known[after]
        if not new.any():
            break
        middle[after[new & known[before]]] = (middle[before] + aim)[new & known[before]]
        middle[before[new & known[after]]] = (middle[after] - aim)[new & known[after]]
        known[before[new]] = known[after[new]] = True
    shares = np.maximum(np.bincount(np.concatenate([before, after]), minlength=store.cells), 1)
    for _ in range(ROUNDS):
        off, move = aim - (middle[after] - middle[before]), np.zeros_like(middle)
        np.add.at(move, after, off)
        np.add.at(move, before, -off)
        middle += 0.5 * move / shares[:, None]
    middle -= middle[held > 0].mean(axis=0)
    middle[joints] = (middle[before] + middle[after]) / 2
    return middle, size, np.column_stack([joints, before, after, axis])


def placed(store, middle: np.ndarray, size: np.ndarray, joints: np.ndarray, span: float) -> wp.array:
    """A mat33 for each own slot, as `shape.py` reads it; a joint with no volumes is left 0."""
    shapes = np.zeros((store.cells, 3, 3), dtype=np.float32)
    shapes[:, 0] = np.stack([middle[:, 1], -middle[:, 0], middle[:, 2]], axis=1) / span
    shapes[:, 1, 0], shapes[:, 2, 2] = size / span, 1  # a volume, unless a joint below
    cell, before, after, axis = joints.T
    reach = np.abs((middle[after] - middle[before])[np.arange(len(cell)), axis])
    shapes[cell, 1] = np.stack([reach, size[before] / 2, size[after] / 2], axis=1) / span
    shapes[cell, 2] = np.stack([0 * axis, axis, -np.ones(len(cell))], axis=1)
    own = store.own_cell.numpy()[:store.count]
    return wp.array(shapes[own], dtype=wp.mat33, device=store.cell_quanta.device)
