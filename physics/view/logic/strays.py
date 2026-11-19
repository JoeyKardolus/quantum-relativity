"""Strays: tracers the fluid carried out of the cells that are drawn are laid in drawn cells again.

While only the flat cut is drawn, a tracer that crosses into a cell off the cut would leave
its streak sticking out of the layer: seen from the side, red dots round a blue slab.
Before each drawing every such tracer, tail and all, is laid at rest in a drawn cell, so
the cut shows only its own fluid from whichever side it is seen.
"""
import numpy as np
import warp as wp

from physics.view.logic.lay_tracers import _move, places
from physics.view.logic.tracers import MARKS, Tracers


def bring_back(store, tracers: Tracers, shown: np.ndarray) -> int:
    """Lay the tracers that lie in own slots not `shown` in shown ones; how many there were."""
    now = tracers.tail[tracers.head].numpy()[:, 1] // store.across
    away = np.flatnonzero(np.ravel(shown)[np.clip(now.astype(int), 0, store.count - 1)] == 0)
    if away.size:
        first = int(store.own_cell[0:1].numpy()[0])
        place = places(store, first, away.size, store.step, store.count, shown, True)
        device = tracers.tail.device
        wp.launch(_move, dim=(MARKS, away.size), device=device,
                  inputs=[wp.array(away.astype(np.int32), dtype=wp.int32, device=device),
                          wp.array(place, dtype=wp.vec3, device=device), tracers.tail])
    return int(away.size)
