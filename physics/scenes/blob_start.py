"""Blob start: an even checkerboard of cells with balls of extra energy in it, each moving.

Contract: `blob_start(cells_per_side, boxes_per_cell_side, blobs, device="cuda")` returns
a physics.potts Universe at rest except for the blobs: new_universe(cells_per_side,
boxes_per_cell_side, seed=1, unevenness=0) (every box holds 1, energy volume and joint
alike, in an empty margin), plus each blob. A blob is a dict with `offset` (3
floats, box units, from the grid's middle (side - 1) / 2; box i's centre at i),
`radius` (boxes; an occupied box is in the ball when its centre is within it),
`energy` (the blob's extra energy in its A boxes, spread evenly over them; its B boxes
get as much per box, as the start's joints do: a joint is a full cell) and `velocity` (3
floats, boxes per time unit): every box in the ball gets momentum = its energy times
that velocity. Nothing outside a cell can hold energy (empty space is no box), so a
blob is a dense, moving region of the foam, not a ball in a vacuum. Overlapping blobs
add their energies; the later blob's velocity wins in shared boxes. Raises ValueError
for a blob that covers no A box or has energy <= 0.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.potts import JOINT, QUANTUM, Universe, new_universe


def blob_start(cells_per_side: int, boxes_per_cell_side: int, blobs: list[dict], device: str = "cuda") -> Universe:
    """An even checkerboard holding balls of extra energy, each moving at its own velocity."""
    even = new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0.0, device=device)
    owner = even.owner.numpy()
    kind = even.kind.numpy()
    energy = even.energy.numpy().astype(np.float64) * QUANTUM
    joint = (owner >= 0) & (kind[np.maximum(owner, 0)] == JOINT)
    side = even.side
    axis = np.arange(side, dtype=np.float64)
    velocity = np.zeros((side, side, side, 3))
    for blob in blobs:
        centre = (side - 1) / 2.0 + np.asarray(blob["offset"], dtype=np.float64)
        inside = (owner >= 0) & (((axis[:, None, None] - centre[0]) ** 2 + (axis[None, :, None] - centre[1]) ** 2
                                  + (axis[None, None, :] - centre[2]) ** 2) <= blob["radius"] ** 2)
        a_boxes = inside & ~joint
        if not a_boxes.any() or blob["energy"] <= 0:
            raise ValueError(f"blob {blob} covers no energy-volume box or holds no energy")
        per_box = blob["energy"] / a_boxes.sum()
        energy[a_boxes] += per_box
        energy[inside & joint] += per_box
        velocity[inside] = np.asarray(blob["velocity"], dtype=np.float64)
    universe = Universe.from_host(owner, energy, kind, boxes_per_cell_side, seed=1, device=device)
    held = universe.energy.numpy().astype(np.float64) * QUANTUM
    universe.momentum.assign(wp.array((held[..., None] * velocity).astype(np.float32), dtype=wp.vec3, device=device))
    return universe
