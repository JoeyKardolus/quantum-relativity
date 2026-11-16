"""View source: what a viewer draws of one universe, one point per occupied Planck box, on the GPU.

Contract: `view_source(state, universe=0)` returns a dict of Warp arrays on the state's
device, one entry per occupied box of that universe (n of them), in its own box units
(the boxes never move, so a box's place is its index):
  positions  vec3 (n,), the box centre (x, y, z) + 0.5
  density    float32 (n,), the box's whole quanta ignoring sign (a box has volume 1)
  kind       int32 (n,), 0 for a plus box (energy volume, red), 1 for a minus box (joint, blue)
  border     int32 (n,), 1 when a face neighbour holds the other sign, is empty, or is
             the wall (a universe that wraps has none) (the boxes that draw the cells' walls), else 0
Boxes come in index order ((x * side + y) * side + z), compacted by a prefix sum on the
GPU; only the count is copied to the host. The simulation as it is: nothing is placed,
smoothed or corrected for the drawing. Raises ValueError for a universe the batch lacks.
"""
from __future__ import annotations

import warp as wp

from physics.engine.face_step import face_step
from physics.engine.reach import reach
from physics.engine.state import State


@wp.kernel
def _mark(quanta: wp.array3d(dtype=wp.int32), low: int, flags: wp.array3d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    flags[ti, tj, tk] = wp.where(quanta[low + ti, tj, tk] != 0, 1, 0)


@wp.kernel
def _gather(quanta: wp.array3d(dtype=wp.int32), side: int, wrap: int, home: int, flags: wp.array3d(dtype=wp.int32),
            places: wp.array3d(dtype=wp.int32), positions: wp.array(dtype=wp.vec3),
            density: wp.array(dtype=wp.float32), kind: wp.array(dtype=wp.int32), border: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    if flags[ti, tj, tk] == 0:
        return
    here = wp.vec3i(home * side + ti, tj, tk)
    held = quanta[here[0], here[1], here[2]]
    slot = places[ti, tj, tk]
    positions[slot] = wp.vec3(float(ti) + 0.5, float(tj) + 0.5, float(tk) + 0.5)
    density[slot] = float(wp.abs(held))
    kind[slot] = wp.where(held > 0, 0, 1)
    edge = int(0)
    for face in range(6):
        there, valid = reach(here, face_step(face), home, side, wrap)
        if not valid:
            edge = 1
        elif quanta[there[0], there[1], there[2]] * held <= 0:
            edge = 1
    border[slot] = edge


def view_source(state: State, universe: int = 0) -> dict:
    """The occupied boxes of one universe: positions, density, kind and border, compacted on the GPU."""
    rows = state.universe(universe)
    device = state.quanta.device
    shape = (state.side,) * 3
    flags = wp.zeros(shape, dtype=wp.int32, device=device)
    places = wp.zeros(shape, dtype=wp.int32, device=device)
    wp.launch(_mark, dim=shape, device=device, inputs=[state.quanta, rows.start, flags])
    wp.utils.array_scan(flags.flatten(), places.flatten(), inclusive=False)
    total = state.side ** 3
    count = int(places.flatten()[total - 1:].numpy()[0]) + int(flags.flatten()[total - 1:].numpy()[0])
    found = {"positions": wp.zeros(count, dtype=wp.vec3, device=device),
             "density": wp.zeros(count, dtype=wp.float32, device=device),
             "kind": wp.zeros(count, dtype=wp.int32, device=device),
             "border": wp.zeros(count, dtype=wp.int32, device=device)}
    if count:
        wp.launch(_gather, dim=shape, device=device,
                  inputs=[state.quanta, state.side, int(state.wrap), universe, flags, places, found["positions"], found["density"],
                          found["kind"], found["border"]])
    return found
