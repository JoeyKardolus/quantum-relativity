"""View source: what a viewer draws of the universe, one point per occupied Planck box, on the GPU.

Contract: `view_source(universe)` returns a dict of Warp arrays on the universe's
device, one entry per occupied box (n of them), in box units (the boxes never move,
so a box's place is its index):
  positions  vec3 (n,), the box centre (x, y, z) + 0.5
  density    float32 (n,), the energy in the box, |psi|**2 (a box has volume 1)
  kind       int32 (n,), its cell's kind, 0 energy volume (A, red), 1 joint (B, blue)
  border     int32 (n,), 1 when a face neighbour belongs to another cell or is empty
             (the boxes that draw the cells' walls), else 0
  cell       int32 (n,), the owning cell's id
Boxes come in index order ((x * side + y) * side + z), compacted by a prefix sum over
the active region (active_region.py; every box outside it is empty) on the GPU;
only the count is copied to the host. The simulation as it is: nothing is
placed, smoothed or corrected for the drawing.
"""
from __future__ import annotations

import warp as wp

from physics.potts.box_owner import box_owner
from physics.potts.direction import direction
from physics.potts.universe import Universe


@wp.kernel
def _mark(owner: wp.array3d(dtype=wp.int32), low: wp.vec3i, dims: wp.vec3i, flags: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    flags[(ti * dims[1] + tj) * dims[2] + tk] = wp.where(owner[ti + low[0], tj + low[1], tk + low[2]] >= 0, 1, 0)


@wp.kernel
def _gather(owner: wp.array3d(dtype=wp.int32), psi: wp.array3d(dtype=wp.vec2d), kind: wp.array(dtype=wp.int32),
            side: int, low: wp.vec3i, dims: wp.vec3i, flags: wp.array(dtype=wp.int32), places: wp.array(dtype=wp.int32),
            positions: wp.array(dtype=wp.vec3), density: wp.array(dtype=wp.float32),
            box_kind: wp.array(dtype=wp.int32), border: wp.array(dtype=wp.int32), cell: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    index = (ti * dims[1] + tj) * dims[2] + tk
    i = ti + low[0]
    j = tj + low[1]
    k = tk + low[2]
    if flags[index] == 0:
        return
    slot = places[index]
    here = owner[i, j, k]
    on_border = 0
    for face in range(6):
        there = wp.vec3i(i, j, k) + direction(face)
        if box_owner(owner, there[0], there[1], there[2], side) != here:
            on_border = 1
    positions[slot] = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    density[slot] = float(wp.dot(psi[i, j, k], psi[i, j, k]))
    box_kind[slot] = kind[here]
    border[slot] = on_border
    cell[slot] = here


def view_source(universe: Universe) -> dict:
    """Positions, energy density, cell kind, border flag and cell of every occupied box, on the GPU."""
    device = universe.owner.device
    side = universe.side
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    low = wp.vec3i(*universe.low)
    total = dims[0] * dims[1] * dims[2]
    flags = wp.zeros(total, dtype=wp.int32, device=device)
    places = wp.zeros(total, dtype=wp.int32, device=device)
    wp.launch(_mark, dim=dims, device=device, inputs=[universe.owner, low, wp.vec3i(*dims), flags])
    wp.utils.array_scan(flags, places, inclusive=False)
    count = int(places[total - 1:].numpy()[0]) + int(flags[total - 1:].numpy()[0])
    found = {"positions": wp.zeros(count, dtype=wp.vec3, device=device),
             "density": wp.zeros(count, dtype=wp.float32, device=device),
             "kind": wp.zeros(count, dtype=wp.int32, device=device),
             "border": wp.zeros(count, dtype=wp.int32, device=device),
             "cell": wp.zeros(count, dtype=wp.int32, device=device)}
    wp.launch(_gather, dim=dims, device=device,
              inputs=[universe.owner, universe.psi, universe.kind, side, low, wp.vec3i(*dims), flags, places,
                      found["positions"],
                      found["density"], found["kind"], found["border"], found["cell"]])
    return found
