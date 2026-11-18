"""Placed: where every own cell of a scene is drawn and how wide, read off the engine.

The engine's squeeze says how far apart a joint's two volumes stand. The volumes are set
on a straight grid, as far apart along each direction as its joints say on the average
(the scenes' joints all say the same); each joint lies between its two volumes.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_5_packing.logic.squeeze import squeeze


def placed(store) -> wp.array:
    """A mat33 for each own slot, as `shape.py` reads it; a joint with no volumes is left 0."""
    squeeze(store)
    cells = store.own_cell.numpy()[:store.count]
    held = store.cell_quanta.numpy()[cells] / (QUANTA_A_BOX * store.across ** 3)
    apart, axis = store.own_squeeze.numpy()[:store.count, 2:].T
    spot = np.stack(np.unravel_index(cells, store.grid), axis=1)[:, :2]  # down, along
    joint = apart > 0
    wide = [apart[joint & (axis == way)].mean() if (joint & (axis == way)).any() else 0.0
            for way in (0, 1)]
    middle = (spot - (np.array(store.grid[:2]) - 1) / 2) / 2 * wide
    shapes = np.zeros((len(cells), 3, 3), dtype=np.float32)
    shapes[:, 0, 0], shapes[:, 0, 1] = middle[:, 1], -middle[:, 0]
    shapes[:, 1, 0], shapes[:, 2, 2] = 1 / np.maximum(held, 1e-9), np.sign(held)
    for slot in np.flatnonzero(joint):
        way = int(axis[slot])
        first, second = (int(np.flatnonzero((spot == spot[slot] + step * np.eye(2, dtype=int)[way])
                                            .all(axis=1))[0]) for step in (-1, 1))
        shapes[slot, 1] = apart[slot], 0.5 / held[first], 0.5 / held[second]
        shapes[slot, 2] = 0, way, 1 / held[slot]
    return wp.array(shapes, dtype=wp.mat33, device=store.cell_quanta.device)
