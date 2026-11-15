"""Region host: a box array's active region (active_region.py) copied to the host as numpy.

Contract: `region_host(universe, array)` returns a numpy copy of array[low:high] over
the universe's active region (a 3D box array of the universe's side: int32, float32,
float64 or vec2d, the box arrays' types),
gathered on the GPU into one contiguous block first so only the region is copied.
Outside the region every box is empty, so nothing occupied is left out.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.potts.universe import Universe


@wp.kernel
def _gather(source: wp.array3d(dtype=wp.int32), low: wp.vec3i, out: wp.array3d(dtype=wp.int32)):
    i, j, k = wp.tid()
    out[i, j, k] = source[i + low[0], j + low[1], k + low[2]]


@wp.kernel
def _gather_wide(source: wp.array3d(dtype=wp.float64), low: wp.vec3i, out: wp.array3d(dtype=wp.float64)):
    i, j, k = wp.tid()
    out[i, j, k] = source[i + low[0], j + low[1], k + low[2]]


@wp.kernel
def _gather_float(source: wp.array3d(dtype=wp.float32), low: wp.vec3i, out: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    out[i, j, k] = source[i + low[0], j + low[1], k + low[2]]


@wp.kernel
def _gather_vector(source: wp.array3d(dtype=wp.vec2d), low: wp.vec3i, out: wp.array3d(dtype=wp.vec2d)):
    i, j, k = wp.tid()
    out[i, j, k] = source[i + low[0], j + low[1], k + low[2]]


_KERNELS = {wp.int32: _gather, wp.float64: _gather_wide, wp.float32: _gather_float, wp.vec2d: _gather_vector}


def region_host(universe: Universe, array: wp.array) -> np.ndarray:
    """The active region of a box array, as numpy."""
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    out = wp.zeros(dims, dtype=array.dtype, device=array.device)
    wp.launch(_KERNELS[array.dtype], dim=dims, device=array.device, inputs=[array, wp.vec3i(*universe.low), out])
    return out.numpy()
