"""Box points: the fluid grid's boxes as drawable points of light, selected and gathered on the GPU.

Contract: `BoxPoints()` keeps GPU buffers between calls. `boxes(a_energy, b_energy,
reference)` takes the two per-box energy arrays (Warp arrays on one device, any
dtype Warp can cast to float32, shape (side^3,) or (side, side, side), box index
(x * side + y) * side + z) and the start reference (`start_reference`, the mean
A and B energy) and returns a `Gathered`: positions (n, 3) float32 box centres in
box units (x + 0.5, y + 0.5, z + 0.5), colours (n, 4) uint8, dense (boxes at or
above CONTRAST_SHOWN) and negative (boxes with an energy below zero).

A box's contrast is (A + B) / (reference[0] + reference[1]), its density over the
start's mean density. Which boxes become points: every box at or above
CONTRAST_SHOWN (the structure), plus a fixed random SAMPLE of the rest (the
background, the same boxes every frame, so it does not flicker). When those
would be more than MOST_POINTS, the dense boxes are thinned by one uniform random
share, never by order or rank (a random share can overshoot by a few points;
those past MOST_POINTS are dropped). A point's light is `opacity(contrast, side)`
(`opacity.py`, fixed log scale, line of sight normalised by the side), divided
by the share of its kind that was kept, so the expected light of the whole grid
is the same as drawing every box; a point whose light codes to 0 is left out.
Colour: with ratios a = A / reference[0] and b = B / reference[1], RGB is
(RED a + BLUE b) / (a + b), so red is pulling (A), blue pushing (B); alpha is the
light as a log byte (`light_code.py`), decoded by the window. The selection runs
on the arrays' device in two passes (count, then gather); the result is copied to
the host once per call (the counts, then the n positions and colours). Point
order is not fixed (additive drawing does not need it). Raises ValueError when
any box energy is not finite; that frame is not drawn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from physics.view.light_code import light_code
from physics.view.opacity import opacity

CONTRAST_SHOWN = 1.5
SAMPLE = 0.01
MOST_POINTS = 3_000_000
RED = (255, 64, 38)
BLUE = (64, 115, 255)
_RED = wp.constant(wp.vec3(*RED))
_BLUE = wp.constant(wp.vec3(*BLUE))
_BACKGROUND_SEED = 7919
_THINNING_SEED = 104729

_DENSE, _BACKGROUND, _BROKEN, _NEGATIVE, _SLOTS = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class Gathered:
    """The drawn boxes of one moment, on the host."""

    positions: np.ndarray
    colours: np.ndarray
    dense: int
    negative: int


@wp.func
def _in_background_sample(box: int) -> bool:
    state = wp.rand_init(_BACKGROUND_SEED, box)
    return wp.randf(state) < wp.float32(SAMPLE)


@wp.kernel
def _count(a_energy: wp.array(dtype=Any), b_energy: wp.array(dtype=Any), mean_density: float,
           counts: wp.array(dtype=wp.int32)):
    box = wp.tid()
    pulling = wp.float32(a_energy[box])
    pushing = wp.float32(b_energy[box])
    if not (wp.isfinite(pulling) and wp.isfinite(pushing)):
        wp.atomic_add(counts, _BROKEN, 1)
        return
    if pulling < 0.0 or pushing < 0.0:
        wp.atomic_add(counts, _NEGATIVE, 1)
    if (pulling + pushing) / mean_density >= wp.float32(CONTRAST_SHOWN):
        wp.atomic_add(counts, _DENSE, 1)
    elif _in_background_sample(box):
        wp.atomic_add(counts, _BACKGROUND, 1)


@wp.kernel
def _gather(a_energy: wp.array(dtype=Any), b_energy: wp.array(dtype=Any), side: int,
            reference_a: float, reference_b: float, dense_kept: float,
            counts: wp.array(dtype=wp.int32), positions: wp.array(dtype=wp.vec3),
            colours: wp.array(dtype=wp.uint8, ndim=2)):
    box = wp.tid()
    pulling = wp.max(wp.float32(a_energy[box]), 0.0)
    pushing = wp.max(wp.float32(b_energy[box]), 0.0)
    contrast = (pulling + pushing) / (reference_a + reference_b)
    kept = wp.float32(1.0)
    if contrast >= wp.float32(CONTRAST_SHOWN):
        if dense_kept < 1.0:
            state = wp.rand_init(_THINNING_SEED, box)
            if wp.randf(state) >= dense_kept:
                return
            kept = dense_kept
    elif _in_background_sample(box):
        kept = wp.float32(SAMPLE)
    else:
        return
    code = wp.round(light_code(opacity(contrast, wp.float32(side)) / kept))
    if code < 1.0:
        return
    slot = wp.atomic_add(counts, 0, 1)
    if slot >= positions.shape[0]:
        return
    column = box // side
    positions[slot] = wp.vec3(float(column // side) + 0.5, float(column % side) + 0.5,
                              float(box % side) + 0.5)
    ratio_a = pulling / reference_a
    ratio_b = pushing / reference_b
    mixed = (_RED * ratio_a + _BLUE * ratio_b) / (ratio_a + ratio_b)
    colours[slot, 0] = wp.uint8(wp.round(mixed[0]))
    colours[slot, 1] = wp.uint8(wp.round(mixed[1]))
    colours[slot, 2] = wp.uint8(wp.round(mixed[2]))
    colours[slot, 3] = wp.uint8(code)


class BoxPoints:
    """Selects and gathers drawn boxes on the GPU into reused buffers and copies them out once."""

    def __init__(self) -> None:
        self._buffers = None

    def boxes(self, a_energy: wp.array, b_energy: wp.array, reference: np.ndarray) -> Gathered:
        """The drawn boxes' centres and colours, on the host."""
        a_flat, b_flat = _flat(a_energy), _flat(b_energy)
        if a_flat.shape != b_flat.shape or a_flat.device != b_flat.device:
            raise ValueError(f"A energy {a_flat.shape} on {a_flat.device} but B energy "
                             f"{b_flat.shape} on {b_flat.device}")
        size = a_flat.shape[0]
        side = round(size ** (1.0 / 3.0))
        if side ** 3 != size:
            raise ValueError(f"{size} boxes is not a cube")
        reference_a, reference_b = (float(value) for value in reference)
        counts, slots, positions, colours = self._buffers_for(min(size, MOST_POINTS), a_flat.device)
        counts.zero_()
        slots.zero_()
        wp.launch(_count, dim=size, device=a_flat.device,
                  inputs=[a_flat, b_flat, reference_a + reference_b, counts])
        dense, background, broken, negative = (int(value) for value in counts.numpy())
        if broken:
            raise ValueError(f"{broken} boxes hold an energy that is not finite")
        dense_kept = _dense_kept(dense, background)
        wp.launch(_gather, dim=size, device=a_flat.device,
                  inputs=[a_flat, b_flat, side, reference_a, reference_b, dense_kept,
                          slots, positions, colours])
        drawn = min(int(slots.numpy()[0]), positions.shape[0])
        if drawn == 0:
            return Gathered(np.zeros((0, 3), np.float32), np.zeros((0, 4), np.uint8), dense, negative)
        positions_host = positions[:drawn].numpy().reshape(drawn, 3)
        colours_host = colours[:drawn].numpy()
        if a_flat.device.is_cpu:
            # .numpy() of a CPU array shares the reused buffer, so the next gather would
            # overwrite a snapshot still being drawn; take an own copy.
            positions_host, colours_host = positions_host.copy(), colours_host.copy()
        return Gathered(positions_host, colours_host, dense, negative)

    def _buffers_for(self, capacity: int, device) -> tuple:
        if self._buffers is None or self._buffers[0] != (capacity, str(device)):
            self._buffers = ((capacity, str(device)),
                             wp.zeros(_SLOTS, dtype=wp.int32, device=device),
                             wp.zeros(1, dtype=wp.int32, device=device),
                             wp.empty(capacity, dtype=wp.vec3, device=device),
                             wp.empty((capacity, 4), dtype=wp.uint8, device=device))
        return self._buffers[1:]


def _dense_kept(dense: int, background: int) -> float:
    """The share of dense boxes kept so dense plus background fits MOST_POINTS."""
    room = MOST_POINTS - background
    if dense <= room:
        return 1.0
    return max(room, 0) / dense


def _flat(array: wp.array) -> wp.array:
    return array if array.ndim == 1 else array.flatten()
