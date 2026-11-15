"""Box points: the occupied Planck boxes as drawable points of light, gathered on the GPU.

Contract: `BoxPoints()` keeps GPU buffers between calls. `boxes(view, reference)`
takes the source's view (`cells_source.py`: `positions` (n, 3) box centres in box
units, as an (n, 3) array of numbers or an (n,) array of 3-vectors; `density` (n,)
energy per box; `kind` (n,) the kind of the cell owning the box, ENERGY_VOLUME or
JOINT; `border` (n,) nonzero where the box touches a box of another cell; Warp
arrays on one device or numpy, which is read on the CPU) and the start reference
(`start_reference`: mean density, boxes deep) and returns a `Gathered`: positions
(m, 3) float32, colours (m, 4) uint8, negative (boxes with a density below zero),
low and high (float32 (3,), the corners of the occupied boxes' bounding box, for
fitting the view; zeros when there are none).

Every occupied box is a candidate; empty boxes are never in the view, so never
drawn. A box's light is `opacity(density / mean density, boxes deep)` (fixed log
scale, `opacity.py`), times BORDER_GAIN on a cell border, so the checkerboard of
cells shows. Past MOST_POINTS boxes one uniform random share is kept (never by
order or rank; a few past MOST_POINTS are dropped) and each kept point's light is
divided by that share, so the expected light of the whole is unchanged. A point
whose light codes to 0 is left out. Colour: RED in an energy volume (A), BLUE in
a joint (B); alpha is the light as a log byte (`light_code.py`), decoded by the
window. Two passes on the view's device (check and bound, then gather); the
result is copied to the host once per call. Point order is not fixed (additive
drawing does not need it). Raises ValueError when the arrays disagree in length
or device, when a density or position is not finite, or a kind is neither
ENERGY_VOLUME nor JOINT; that frame is not drawn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from physics.view.light_code import light_code
from physics.view.opacity import opacity

ENERGY_VOLUME = 0
JOINT = 1
BORDER_GAIN = 2.0
MOST_POINTS = 3_000_000
RED = (255, 64, 38)
BLUE = (64, 115, 255)
_RED = wp.constant(wp.vec3(*RED))
_BLUE = wp.constant(wp.vec3(*BLUE))
_THINNING_SEED = 104729
_HUGE = 3.0e38

_BROKEN, _NEGATIVE, _BAD_KIND, _SLOTS = 0, 1, 2, 3


@dataclass(frozen=True)
class Gathered:
    """The drawn boxes of one moment, on the host."""

    positions: np.ndarray
    colours: np.ndarray
    negative: int
    low: np.ndarray
    high: np.ndarray


@wp.kernel
def _rows(vectors: wp.array(dtype=Any), rows: wp.array(dtype=wp.float32, ndim=2)):
    box = wp.tid()
    vector = vectors[box]
    for axis in range(3):
        rows[box, axis] = wp.float32(vector[axis])


@wp.kernel
def _check(positions: wp.array(dtype=Any, ndim=2), density: wp.array(dtype=Any),
           kind: wp.array(dtype=Any), counts: wp.array(dtype=wp.int32),
           low: wp.array(dtype=wp.float32), high: wp.array(dtype=wp.float32)):
    box = wp.tid()
    energy = wp.float32(density[box])
    finite = wp.isfinite(energy)
    for axis in range(3):
        coordinate = wp.float32(positions[box, axis])
        if not wp.isfinite(coordinate):
            finite = False
        else:
            wp.atomic_min(low, axis, coordinate)
            wp.atomic_max(high, axis, coordinate)
    if not finite:
        wp.atomic_add(counts, _BROKEN, 1)
    elif energy < 0.0:
        wp.atomic_add(counts, _NEGATIVE, 1)
    owner_kind = wp.int32(kind[box])
    if owner_kind != ENERGY_VOLUME and owner_kind != JOINT:
        wp.atomic_add(counts, _BAD_KIND, 1)


@wp.kernel
def _gather(positions: wp.array(dtype=Any, ndim=2), density: wp.array(dtype=Any),
            kind: wp.array(dtype=Any), border: wp.array(dtype=Any), mean_density: float,
            boxes_deep: float, kept: float, slots: wp.array(dtype=wp.int32),
            points: wp.array(dtype=wp.vec3), colours: wp.array(dtype=wp.uint8, ndim=2)):
    box = wp.tid()
    if kept < 1.0:
        state = wp.rand_init(_THINNING_SEED, box)
        if wp.randf(state) >= kept:
            return
    contrast = wp.max(wp.float32(density[box]), 0.0) / mean_density
    light = opacity(contrast, wp.float32(boxes_deep)) / kept
    if wp.int32(border[box]) != 0:
        light = light * wp.float32(BORDER_GAIN)
    code = wp.round(light_code(light))
    if code < 1.0:
        return
    slot = wp.atomic_add(slots, 0, 1)
    if slot >= points.shape[0]:
        return
    points[slot] = wp.vec3(wp.float32(positions[box, 0]), wp.float32(positions[box, 1]),
                           wp.float32(positions[box, 2]))
    tint = _BLUE
    if wp.int32(kind[box]) == ENERGY_VOLUME:
        tint = _RED
    colours[slot, 0] = wp.uint8(tint[0])
    colours[slot, 1] = wp.uint8(tint[1])
    colours[slot, 2] = wp.uint8(tint[2])
    colours[slot, 3] = wp.uint8(code)


class BoxPoints:
    """Gathers the occupied boxes as points into reused GPU buffers and copies them out once."""

    def __init__(self) -> None:
        self._buffers = None

    def boxes(self, view, reference) -> Gathered:
        """The drawn boxes' centres and colours, and the occupied bounds, on the host."""
        density = _warp(view.density)
        device = density.device
        positions, kind, border = (_warp(array, device) for array in (view.positions, view.kind, view.border))
        positions = _as_rows(positions)
        size = density.shape[0]
        for name, array in (("positions", positions), ("kind", kind), ("border", border)):
            if array.shape[0] != size or array.device != device:
                raise ValueError(f"{name}: {array.shape[0]} boxes on {array.device}, "
                                 f"density: {size} on {device}")
        mean_density, boxes_deep = (float(value) for value in reference)
        counts, slots, low, high, points, colours = self._buffers_for(min(size, MOST_POINTS), device)
        counts.zero_()
        slots.zero_()
        low.fill_(_HUGE)
        high.fill_(-_HUGE)
        if size == 0:
            return _nothing(0, np.zeros(3, np.float32), np.zeros(3, np.float32))
        wp.launch(_check, dim=size, device=device, inputs=[positions, density, kind, counts, low, high])
        broken, negative, bad_kind = (int(value) for value in counts.numpy())
        if broken:
            raise ValueError(f"{broken} boxes hold a density or position that is not finite")
        if bad_kind:
            raise ValueError(f"{bad_kind} boxes belong to a cell kind that is neither "
                             f"{ENERGY_VOLUME} (energy volume) nor {JOINT} (joint)")
        kept = 1.0 if size <= MOST_POINTS else MOST_POINTS / size
        wp.launch(_gather, dim=size, device=device,
                  inputs=[positions, density, kind, border, mean_density, boxes_deep, kept,
                          slots, points, colours])
        low_host, high_host = low.numpy().copy(), high.numpy().copy()
        drawn = min(int(slots.numpy()[0]), points.shape[0])
        if drawn == 0:
            return _nothing(negative, low_host, high_host)
        points_host = points[:drawn].numpy().reshape(drawn, 3)
        colours_host = colours[:drawn].numpy()
        if device.is_cpu:
            # .numpy() of a CPU array shares the reused buffer, so the next gather would
            # overwrite a snapshot still being drawn; take an own copy.
            points_host, colours_host = points_host.copy(), colours_host.copy()
        return Gathered(points_host, colours_host, negative, low_host, high_host)

    def _buffers_for(self, capacity: int, device) -> tuple:
        if self._buffers is None or self._buffers[0] != (capacity, str(device)):
            self._buffers = ((capacity, str(device)),
                             wp.zeros(_SLOTS, dtype=wp.int32, device=device),
                             wp.zeros(1, dtype=wp.int32, device=device),
                             wp.zeros(3, dtype=wp.float32, device=device),
                             wp.zeros(3, dtype=wp.float32, device=device),
                             wp.empty(max(capacity, 1), dtype=wp.vec3, device=device),
                             wp.empty((max(capacity, 1), 4), dtype=wp.uint8, device=device))
        return self._buffers[1:]


def _nothing(negative: int, low: np.ndarray, high: np.ndarray) -> Gathered:
    return Gathered(np.zeros((0, 3), np.float32), np.zeros((0, 4), np.uint8), negative, low, high)


def _warp(array, device=None) -> wp.array:
    """A Warp array as is; numpy (bool as uint8) onto the given device, or the CPU."""
    if isinstance(array, wp.array):
        return array
    host = np.asarray(array)
    if host.dtype == np.bool_:
        host = host.astype(np.uint8)
    return wp.array(host, device=device or "cpu")


def _as_rows(positions: wp.array) -> wp.array:
    """Positions as an (n, 3) array: an (n,) array of 3-vectors is copied to float32 rows."""
    if positions.ndim == 2:
        return positions
    rows = wp.empty((positions.shape[0], 3), dtype=wp.float32, device=positions.device)
    if positions.shape[0]:
        wp.launch(_rows, dim=positions.shape[0], device=positions.device, inputs=[positions, rows])
    return rows
