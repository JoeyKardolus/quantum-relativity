"""Placed: where every own cell is drawn and how wide, read off the sizes and the joints.

A cell has no place of its own, so the picture's places are worked out from what the engine
holds: each joint says how far apart its two volumes stand along its axis (the squeeze's
`apart`). `settle` finds the volumes' middles that agree best with all the joints at once
(they cannot all agree: sizes change, space is not a lattice); a new volume starts where
its steps in the pattern put it, at the joints' usual distance a step. A volume is drawn
as wide as its size, a joint between its two volumes. The picture is drawn `span` times
smaller than the cells are.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_5_packing.logic.squeeze import squeeze

ROUNDS = 40  # times the middles are moved toward what their joints say, a frame


def settle(store, middle: np.ndarray) -> tuple:
    """Every place's middle (places, 3; not a number: no volume yet) from the last ones, each
    cell's size, the joints (cell, before, after, axis) and the own volumes."""
    squeeze(store)
    held, own = store.cell_quanta.numpy(), store.own_cell.numpy()[:store.count]
    size = QUANTA_A_BOX * store.across ** 3 / np.maximum(np.abs(held), 1)
    felt = store.own_squeeze.numpy()[:store.count]
    lies = felt[:, 2] > 0  # the joints that lie between two volumes
    joints, axis, apart = own[lies], felt[lies, 3].astype(int), felt[lies, 2]
    before, after = joints - store.side ** (2 - axis), joints + store.side ** (2 - axis)
    aim = np.zeros((len(joints), 3))  # from the volume before to the one after
    aim[np.arange(len(joints)), axis] = apart
    volumes = own[held[own] > 0]
    if len(middle) != held.size:
        middle = np.full((held.size, 3), np.nan)
    new = volumes[np.isnan(middle[volumes, 0])]
    a_step = np.median(apart) / 2 if len(joints) else 0.0
    middle[new] = (steps_of(new, store.side) - steps_of(volumes[:1], store.side)) * a_step
    number = np.full(held.size, -1)
    number[volumes] = np.arange(len(volumes))  # the relaxing visits the volumes only
    first, second, spot = number[before], number[after], middle[volumes]
    shares = np.maximum(np.bincount(np.concatenate([first, second]), minlength=len(volumes)), 1)
    for _ in range(ROUNDS):
        off, move = aim - (spot[second] - spot[first]), np.zeros_like(spot)
        np.add.at(move, second, off)
        np.add.at(move, first, -off)
        spot += 0.5 * move / shares[:, None]
    middle[volumes] = spot - spot.mean(axis=0)
    return middle, size, np.column_stack([joints, before, after, axis]), volumes


def placed(store, middle: np.ndarray, size: np.ndarray, joints: np.ndarray, span: float,
           shown=1) -> wp.array:
    """A mat33 for each own slot, as `shape.py` reads it; a slot not `shown` counts 0: not drawn."""
    shapes = np.zeros((len(size), 3, 3), dtype=np.float32)
    shapes[:, 0] = np.nan_to_num(np.stack([middle[:, 1], -middle[:, 0], middle[:, 2]], axis=1)) / span
    shapes[:, 1, 0], shapes[:, 2, 2] = size / span, np.sign(store.cell_quanta.numpy())
    cell, before, after, axis = joints.T
    reach = np.abs((middle[after] - middle[before])[np.arange(len(cell)), axis])
    shapes[cell, 0] = (shapes[before, 0] + shapes[after, 0]) / 2
    shapes[cell, 1] = np.stack([reach, size[before] / 2, size[after] / 2], axis=1) / span
    shapes[cell, 2] = np.stack([0 * axis, axis, -np.ones(len(cell))], axis=1)
    own = store.own_cell.numpy()[:store.count]
    shapes = shapes[own]
    shapes[:, 2, 2] *= np.ravel(shown)
    return wp.array(shapes, dtype=wp.mat33, device=store.cell_quanta.device)
