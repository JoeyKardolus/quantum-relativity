"""Blob start: an even start of wrapped energy volumes with balls of extra whole quanta in it, each moving.

Contract: `blob_start(cells_per_side, boxes_per_cell_side, blobs, device="cuda")` returns
a physics.engine State at rest except for the blobs: new_universe(cells_per_side,
boxes_per_cell_side, seed=1, unevenness=0) (every box holds BOX_QUANTA, 1000 quanta:
+ in an energy volume's box, - in a joint's, inside an empty margin), plus each blob.
A blob is a dict with `offset` (3 floats, box units, from the grid's middle (side - 1)
/ 2; box i's centre at i), `radius` (boxes; an occupied box is in the ball when its
centre is within it), `energy` (the blob's extra energy in units of BOX_QUANTA, spread
evenly over its plus boxes; its minus boxes get as many extra quanta per box, as the
start's joints hold as many as its volumes: a joint is a full cell) or, in its place,
`density` (the extra energy per box in those units, plus and minus boxes alike), and
`velocity` (3 floats, boxes per time unit). Each box's extra is rounded to whole
quanta and keeps the box's sign. The fluid is sqrt(|quanta|) exp(i phase) and its flow
is the phase's gradient (hbar = mass = 1), so a ball moves when its phase rises along
the velocity: phase = velocity . (x - centre) inside the ball, 0 outside it. The phase
jumps at the ball's edge by up to |velocity| radius, so that product should stay well
under pi. Nothing outside the foam holds quanta (zero is no space), so a blob is a
dense, moving region of the foam, not a ball in a vacuum. Overlapping blobs add their
quanta; the later blob's phase wins in shared boxes. Raises ValueError for a blob that
covers no plus box or has energy (or density) <= 0.
"""
from __future__ import annotations

import numpy as np

from physics.engine import State, host_boxes, lattice_of, new_universe

BOX_QUANTA = 1000


def blob_start(cells_per_side: int, boxes_per_cell_side: int, blobs: list[dict], device: str = "cuda") -> State:
    """An even foam of whole quanta holding balls of extra quanta, each moving at its own velocity."""
    even = new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0.0, box_quanta=BOX_QUANTA,
                        device=device)
    quanta = host_boxes(even)["quanta"].astype(np.int64)
    side = even.side
    places = np.stack(np.meshgrid(*[np.arange(side, dtype=np.float64)] * 3, indexing="ij"), axis=-1)
    phase = np.zeros((side, side, side))
    extra = np.zeros(quanta.shape)
    for blob in blobs:
        offsets = places - ((side - 1) / 2.0 + np.asarray(blob["offset"], dtype=np.float64))
        inside = (quanta != 0) & ((offsets ** 2).sum(axis=-1) <= blob["radius"] ** 2)
        plus_boxes = inside & (quanta > 0)
        per_box = blob["density"] if "density" in blob else blob["energy"] / max(int(plus_boxes.sum()), 1)
        if not plus_boxes.any() or per_box <= 0:
            raise ValueError(f"blob {blob} covers no energy-volume box or holds no energy")
        extra[inside] += per_box * BOX_QUANTA
        phase[inside] = (offsets @ np.asarray(blob.get("velocity", (0.0, 0.0, 0.0)), dtype=np.float64))[inside]
    quanta = quanta + np.sign(quanta) * np.rint(extra).astype(np.int64)
    return State.from_host(quanta.astype(np.int32), phase, seed=1, device=device, lattice=lattice_of(even))
