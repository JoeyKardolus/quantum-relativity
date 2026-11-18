"""Placed: where every own cell of a scene is drawn and how wide, read off the engine.

The engine's squeeze says how far apart a joint's two volumes stand. The volumes are set
on a straight grid, as far apart along each direction as its joints say on the average;
each joint lies between its two volumes, and the half of a joint past the last volume is
drawn before the first, where the grid wraps round to. `layer` is the layer of volumes
along the third direction that is drawn at the picture's middle.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_5_packing.logic.squeeze import squeeze


def placed(store, layer: float = None) -> wp.array:
    """A mat33 for each own slot, as `shape.py` reads it; a joint with no volumes is left 0."""
    squeeze(store)
    cells = store.own_cell.numpy()[:store.count]
    held = store.cell_quanta.numpy() / (QUANTA_A_BOX * store.across ** 3)
    apart, axis = store.own_squeeze.numpy()[:store.count, 2:].T
    axis = axis.astype(int)
    spot = np.stack(np.unravel_index(cells, store.grid), axis=1)  # places: down, along, deep
    joint, reds = apart > 0, (np.array(store.grid) + 1) // 2
    wide = np.array([apart[joint & (axis == way)].mean() if (joint & (axis == way)).any() else 0.0
                     for way in range(3)])
    first = (reds - 1) / 2 if layer is None else np.array([*(reds[:2] - 1) / 2, layer])
    middle = (spot / 2 - first) * wide
    shapes = np.zeros((len(cells), 3, 3), dtype=np.float32)
    shapes[:, 0] = np.stack([middle[:, 1], -middle[:, 0], middle[:, 2]], axis=1)
    shapes[:, 1, 0], shapes[:, 2, 2] = 1 / np.maximum(held[cells], 1e-9), np.sign(held[cells])
    joint = np.flatnonzero(joint)
    way, at = axis[joint], cells[joint]
    step, place = np.array([store.grid[1] * store.grid[2], store.grid[2], 1])[way], spot[joint, way]
    wraps = place == np.array(store.grid)[way] - 1  # past the last volume: the first is after it
    after = np.where(wraps, at - place * step, at + step)
    shapes[joint, 1] = np.stack([apart[joint], 0.5 / held[at - step], 0.5 / held[after]], axis=1)
    shapes[joint, 2] = np.stack([wraps * -(reds * wide)[way], way, -np.ones(len(joint))], axis=1)
    return wp.array(shapes, dtype=wp.mat33, device=store.cell_quanta.device)
