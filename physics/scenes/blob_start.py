"""Blob start: an even start of wrapped cells with balls of extra energy in it, each moving.

Contract: `blob_start(cells_per_side, boxes_per_cell_side, blobs, device="cuda")` returns
a physics.potts Universe at rest except for the blobs: new_universe(cells_per_side,
boxes_per_cell_side, seed=1, unevenness=0) (every box holds 1, energy volume and joint
alike, in an empty margin), plus each blob. A blob is a dict with `offset` (3
floats, box units, from the grid's middle (side - 1) / 2; box i's centre at i),
`radius` (boxes; an occupied box is in the ball when its centre is within it),
`energy` (the blob's extra energy in its A boxes, spread evenly over them; its B boxes
get as much per box, as the start's joints do: a joint is a full cell) or, in its
place, `density` (the extra energy per box, A and B alike) and `velocity` (3
floats, boxes per time unit). The fluid is one complex number per box, psi =
sqrt(energy) exp(i phase), and its flow is the phase's gradient (hbar = mass = 1), so a
ball moves when its phase rises along the velocity: phase = velocity . (x - centre)
inside the ball, 0 outside it (Universe.from_host's `phase`). The phase jumps at the
ball's edge by up to |velocity| radius, so that product should stay well under pi.
Nothing outside a cell can hold energy (empty space is no box), so a blob is a dense,
moving region of the foam, not a ball in a vacuum. Overlapping blobs add their
energies; the later blob's phase wins in shared boxes. Raises ValueError for a blob
that covers no A box or has energy (or density) <= 0.
"""
from __future__ import annotations

import numpy as np

from physics.potts import JOINT, Universe, new_universe
from physics.scenes.wave_field import wave_field


def blob_start(cells_per_side: int, boxes_per_cell_side: int, blobs: list[dict], device: str = "cuda") -> Universe:
    """An even checkerboard holding balls of extra energy, each moving at its own velocity."""
    even = new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0.0, device=device)
    owner = even.owner.numpy()
    kind = even.kind.numpy()
    energy = np.abs(wave_field(even)) ** 2
    joint = (owner >= 0) & (kind[np.maximum(owner, 0)] == JOINT)
    side = even.side
    places = np.stack(np.meshgrid(*[np.arange(side, dtype=np.float64)] * 3, indexing="ij"), axis=-1)
    phase = np.zeros((side, side, side))
    for blob in blobs:
        offsets = places - ((side - 1) / 2.0 + np.asarray(blob["offset"], dtype=np.float64))
        inside = (owner >= 0) & ((offsets ** 2).sum(axis=-1) <= blob["radius"] ** 2)
        a_boxes = inside & ~joint
        per_box = blob["density"] if "density" in blob else blob["energy"] / max(int(a_boxes.sum()), 1)
        if not a_boxes.any() or per_box <= 0:
            raise ValueError(f"blob {blob} covers no energy-volume box or holds no energy")
        energy[a_boxes] += per_box
        energy[inside & joint] += per_box
        phase[inside] = (offsets @ np.asarray(blob.get("velocity", (0.0, 0.0, 0.0)), dtype=np.float64))[inside]
    return Universe.from_host(owner, energy, kind, boxes_per_cell_side, seed=1, device=device, phase=phase)
