"""Placed: where every own cell is drawn and how wide, read off the sizes and the joints.

A cell has no place of its own, so the picture's places are worked out from what the engine
holds: each joint says how far apart its two volumes stand (the squeeze's `apart`).
`settle` finds the volumes' middles that put every joint's two volumes that far apart
(`fit.py`); a new volume starts at the joint's distance from a neighbour, along the
pattern. A volume is drawn as wide as its size, a joint from the one of its volumes to the
other, leaning as they lie. Everything is drawn `span` times smaller than it is: one scale
for every distance and every width in a frame.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_5_packing.logic.squeeze import squeeze
from physics.view.logic.fit import begun, fitted

OTHERS = np.array([[1, 2], [0, 2], [0, 1]])  # the two directions a joint does not lie along


def joined(store) -> tuple:
    """Each cell's size, the joints that lie between two volumes, standing or collapsed (cell,
    before, after, axis), the own volumes, and how far apart each joint's two volumes stand."""
    squeeze(store)
    held, own = store.cell_quanta.numpy(), store.own_cell.numpy()[:store.count]
    size = np.where(held == 0, 0.0, QUANTA_A_BOX * store.across ** 3 / np.maximum(np.abs(held), 1))
    felt = store.own_squeeze.numpy()[:store.count]
    lies = felt[:, 2] > 0
    joints, axis = own[lies], felt[lies, 3].astype(int)
    ends = [joints + way * store.side ** (2 - axis) for way in (-1, 1)]
    return size, np.column_stack([joints, *ends, axis]), own[(held[own] > 0) & ~lies], felt[lies, 2]


def settle(store, middle: np.ndarray) -> tuple:
    """Every place's middle (places, 3; not a number: no volume yet) from the last ones, then
    what `joined` gives."""
    size, joints, volumes, apart = joined(store)
    aim = np.zeros((len(joints), 3))  # from the volume before to the one after, along the pattern
    aim[np.arange(len(joints)), joints[:, 3]] = apart
    if len(middle) != len(size):
        middle = np.full((len(size), 3), np.nan)
    number = np.full(len(size), -1)
    number[volumes] = np.arange(len(volumes))  # the fit visits the volumes only
    first, second, spot = number[joints[:, 1]], number[joints[:, 2]], middle[volumes]
    spot[:1] = np.nan_to_num(spot[:1])  # the first volume: the middle
    spot = begun(spot, first, second, aim)
    alone = np.isnan(spot[:, 0])  # no joint yet: at its steps in the pattern
    a_step = np.median(apart) / 2 if len(joints) else 0.0
    spot[alone] = spot[0] + (steps_of(volumes[alone], store.side) - steps_of(volumes[:1], store.side)) * a_step
    spot = fitted(spot, first, second, apart, aim)
    middle[volumes] = spot - spot.mean(axis=0)
    return middle, size, joints, volumes, apart


def placed(store, middle: np.ndarray, size: np.ndarray, joints: np.ndarray, span: float,
           shown=1) -> wp.array:
    """A mat33 for each own slot, as `shape.py` reads it; a slot not `shown` counts 0: not drawn."""
    shapes = np.zeros((len(size), 3, 3), dtype=np.float32)
    shapes[:, 0] = np.nan_to_num(np.stack([middle[:, 1], -middle[:, 0], middle[:, 2]], axis=1)) / span
    shapes[:, 1, 0], shapes[:, 2, 2] = size / span, np.sign(store.cell_quanta.numpy())
    cell, before, after, axis = joints.T
    gap = middle[after] - middle[before]
    reach = np.take_along_axis(gap, axis[:, None], 1)  # along the joint's direction; never
    reach = np.maximum(reach, 0.2 * np.linalg.norm(gap, axis=1, keepdims=True) + 1e-9)  # flat
    lean = np.take_along_axis(gap / reach, OTHERS[axis], 1)  # sideways, a step along it
    shapes[cell, 0] = (shapes[before, 0] + shapes[after, 0]) / 2
    shapes[cell, 1] = np.column_stack([reach[:, 0], size[before] / 2, size[after] / 2]) / span
    shapes[cell, 2] = np.column_stack([lean, -1.0 - axis])
    fell = cell[store.cell_quanta.numpy()[cell] > 0]  # a collapsed joint: a red ball between its two
    shapes[fell, 1], shapes[fell, 2] = np.column_stack([size[fell], 0 * fell, 0 * fell]) / span, (0, 0, 1)
    shapes = shapes[store.own_cell.numpy()[:store.count]]
    shapes[:, 2, 2] *= np.ravel(shown)
    return wp.array(shapes, dtype=wp.mat33, device=store.cell_quanta.device)
