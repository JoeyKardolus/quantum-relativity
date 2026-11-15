"""Box points: the occupied Planck boxes as drawable points of light, gathered on the GPU.

Contract: `BoxPoints()` keeps GPU buffers between calls. `boxes(view, reference,
shown=SHOW_BOTH)` takes the source's view (`cells_source.py`: `positions` (n, 3) box centres in box
units, as an (n, 3) array of numbers or an (n,) array of 3-vectors; `density` (n,)
energy per box; `kind` (n,) the kind of the cell owning the box, ENERGY_VOLUME or
JOINT; `border` (n,) nonzero where the box touches a box of another cell; Warp
arrays on one device or numpy, which is read on the CPU) and the start reference
(`start_reference`: the energy volumes' mean density, the joints' mean density,
boxes deep) and `shown`, which kinds to draw (SHOW_ENERGY_VOLUMES, SHOW_JOINTS or
SHOW_BOTH, the default), and returns a `Gathered`: positions
(m, 3) float32, colours (m, 4) uint8, negative (boxes with a density below zero),
low and high (float32 (3,), the corners of the occupied boxes' bounding box, for
fitting the view, over every occupied box whatever is shown; zeros when there are
none).

Every occupied box is a candidate; empty boxes are never in the view, so never
drawn, and a box of a kind not shown is left out. A box's light is
`opacity(density / its own kind's start mean density, boxes deep)` (fixed log
scale, `opacity.py`), so each kind has its own fixed contrast scale and a joint at
its kind's mean is as bright as an energy volume at its own; times BORDER_GAIN on
a cell border, so cells read as cells. Past MOST_POINTS shown boxes one uniform
random share is kept (never by
order or rank; a few past MOST_POINTS are dropped) and each kept point's light is
divided by that share, so the expected light of the whole is unchanged. A point
whose light codes to 0 is left out. Colour: RED in an energy volume (A), BLUE in
a joint (B); alpha is the light as a log byte (`light_code.py`), decoded by the
window. Two passes on the view's device (check and bound, then gather); the
result is copied to the host once per call. Point order is not fixed (additive
drawing does not need it). Raises ValueError when the arrays disagree in length
or device, when a density or position is not finite, or a kind is neither
ENERGY_VOLUME nor JOINT, or `shown` is none of the three; that frame is not drawn.
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
SHOW_ENERGY_VOLUMES = 1
SHOW_JOINTS = 2
SHOW_BOTH = SHOW_ENERGY_VOLUMES | SHOW_JOINTS
RED = (255, 64, 38)
BLUE = (64, 115, 255)
_RED = wp.constant(wp.vec3(*RED))
_BLUE = wp.constant(wp.vec3(*BLUE))
_THINNING_SEED = 104729
_HUGE = 3.0e38

_BROKEN, _NEGATIVE, _BAD_KIND, _SHOWN, _SLOTS = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class Gathered:
    """The drawn boxes of one moment, on the host."""

    positions: np.ndarray
    colours: np.ndarray
    negative: int
    low: np.ndarray
    high: np.ndarray


@wp.func
def _is_shown(owner_kind: wp.int32, shown: wp.int32) -> bool:
    wanted = SHOW_JOINTS
    if owner_kind == ENERGY_VOLUME:
        wanted = SHOW_ENERGY_VOLUMES
    return (shown & wanted) != 0


@wp.kernel
def _rows(vectors: wp.array(dtype=Any), rows: wp.array(dtype=wp.float32, ndim=2)):
    box = wp.tid()
    vector = vectors[box]
    for axis in range(3):
        rows[box, axis] = wp.float32(vector[axis])


@wp.kernel
def _check(positions: wp.array(dtype=Any, ndim=2), density: wp.array(dtype=Any),
           kind: wp.array(dtype=Any), shown: wp.int32, counts: wp.array(dtype=wp.int32),
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
    elif _is_shown(owner_kind, shown):
        wp.atomic_add(counts, _SHOWN, 1)


@wp.kernel
def _gather(positions: wp.array(dtype=Any, ndim=2), density: wp.array(dtype=Any),
            kind: wp.array(dtype=Any), border: wp.array(dtype=Any), volume_mean: float,
            joint_mean: float, boxes_deep: float, shown: wp.int32, kept: float,
            slots: wp.array(dtype=wp.int32),
            points: wp.array(dtype=wp.vec3), colours: wp.array(dtype=wp.uint8, ndim=2)):
    box = wp.tid()
    owner_kind = wp.int32(kind[box])
    if not _is_shown(owner_kind, shown):
        return
    if kept < 1.0:
        state = wp.rand_init(_THINNING_SEED, box)
        if wp.randf(state) >= kept:
            return
    mean_density = wp.float32(joint_mean)
    if owner_kind == ENERGY_VOLUME:
        mean_density = wp.float32(volume_mean)
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
    if owner_kind == ENERGY_VOLUME:
        tint = _RED
    colours[slot, 0] = wp.uint8(tint[0])
    colours[slot, 1] = wp.uint8(tint[1])
    colours[slot, 2] = wp.uint8(tint[2])
    colours[slot, 3] = wp.uint8(code)


class BoxPoints:
    """Gathers the occupied boxes as points into reused GPU buffers and copies them out once."""

    def __init__(self) -> None:
        self._buffers = None

    def boxes(self, view, reference, shown: int = SHOW_BOTH) -> Gathered:
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
        volume_mean, joint_mean, boxes_deep = (float(value) for value in reference)
        if shown not in (SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH):
            raise ValueError(f"shown: {shown}, expected {SHOW_ENERGY_VOLUMES} (energy volumes), "
                             f"{SHOW_JOINTS} (joints) or {SHOW_BOTH} (both)")
        counts, slots, low, high, points, colours = self._buffers_for(min(size, MOST_POINTS), device)
        counts.zero_()
        slots.zero_()
        low.fill_(_HUGE)
        high.fill_(-_HUGE)
        if size == 0:
            return _nothing(0, np.zeros(3, np.float32), np.zeros(3, np.float32))
        wp.launch(_check, dim=size, device=device, inputs=[positions, density, kind, int(shown), counts, low, high])
        broken, negative, bad_kind, shown_count = (int(value) for value in counts.numpy())
        if broken:
            raise ValueError(f"{broken} boxes hold a density or position that is not finite")
        if bad_kind:
            raise ValueError(f"{bad_kind} boxes belong to a cell kind that is neither "
                             f"{ENERGY_VOLUME} (energy volume) nor {JOINT} (joint)")
        kept = 1.0 if shown_count <= MOST_POINTS else MOST_POINTS / shown_count
        wp.launch(_gather, dim=size, device=device,
                  inputs=[positions, density, kind, border, volume_mean, joint_mean, boxes_deep,
                          int(shown), kept, slots, points, colours])
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
