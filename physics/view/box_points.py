"""Box points: the occupied boxes of the fluid grid as drawable points, gathered on the GPU.

Contract: `BoxPoints()` keeps GPU buffers between calls. `boxes(a_energy, b_energy,
reference)` takes the two per-box energy arrays (Warp arrays on one device, any
dtype Warp can cast to float32, shape (side^3,) or (side, side, side), box index
(x * side + y) * side + z) and the start reference (`start_reference`, two
values) and returns a `Gathered`: positions (n, 3) float32 box centres in box
units (x + 0.5, y + 0.5, z + 0.5), colours (n, 4) uint8 and negative (boxes with
an energy below zero, drawn by their positive part only). A box is shown when its
A energy is above SHOWN x reference[0] or its B energy above SHOWN x reference[1].
Its colour holds both kinds in one point: with a = opacity(A / reference[0]) and
b = opacity(B / reference[1]) (`opacity.py`, fixed log scale; a kind at or below
its SHOWN line adds nothing, so a box of pure A is pure red), alpha is
min(a + b, 1) and RGB is (RED a + BLUE b) / (a + b), so additive drawing
(src alpha, one) gives RED a + BLUE b. The gather runs on the arrays' device;
the result is copied to the host once per call (a count, then the n positions and
colours). Point order is not fixed (additive drawing does not need it). Raises
ValueError when any box energy is not finite; that frame is not drawn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from physics.view.opacity import opacity

SHOWN = 0.01
RED = (255, 64, 38)
BLUE = (64, 115, 255)
_RED = wp.constant(wp.vec3(*RED))
_BLUE = wp.constant(wp.vec3(*BLUE))


@dataclass(frozen=True)
class Gathered:
    """The shown boxes of one moment, on the host."""

    positions: np.ndarray
    colours: np.ndarray
    negative: int


@wp.kernel
def _gather(a_energy: wp.array(dtype=Any), b_energy: wp.array(dtype=Any), side: int,
            reference_a: float, reference_b: float, shown_a: float, shown_b: float,
            counts: wp.array(dtype=wp.int32), positions: wp.array(dtype=wp.vec3),
            colours: wp.array(dtype=wp.uint8, ndim=2)):
    box = wp.tid()
    pulling = wp.float32(a_energy[box])
    pushing = wp.float32(b_energy[box])
    if not (wp.isfinite(pulling) and wp.isfinite(pushing)):
        wp.atomic_add(counts, 1, 1)
        return
    if pulling < 0.0 or pushing < 0.0:
        wp.atomic_add(counts, 2, 1)
    if pulling <= shown_a and pushing <= shown_b:
        return
    slot = wp.atomic_add(counts, 0, 1)
    column = box // side
    positions[slot] = wp.vec3(float(column // side) + 0.5, float(column % side) + 0.5,
                              float(box % side) + 0.5)
    solid_a = wp.float32(0.0)
    solid_b = wp.float32(0.0)
    if pulling > shown_a:
        solid_a = opacity(pulling / reference_a)
    if pushing > shown_b:
        solid_b = opacity(pushing / reference_b)
    total = solid_a + solid_b
    mixed = (_RED * solid_a + _BLUE * solid_b) / total
    colours[slot, 0] = wp.uint8(wp.round(mixed[0]))
    colours[slot, 1] = wp.uint8(wp.round(mixed[1]))
    colours[slot, 2] = wp.uint8(wp.round(mixed[2]))
    colours[slot, 3] = wp.uint8(wp.round(wp.min(total, 1.0) * 255.0))


class BoxPoints:
    """Gathers shown boxes on the GPU into reused buffers and copies them out once."""

    def __init__(self) -> None:
        self._buffers = None

    def boxes(self, a_energy: wp.array, b_energy: wp.array, reference: np.ndarray) -> Gathered:
        """The shown boxes' centres and colours, on the host."""
        a_flat, b_flat = _flat(a_energy), _flat(b_energy)
        if a_flat.shape != b_flat.shape or a_flat.device != b_flat.device:
            raise ValueError(f"A energy {a_flat.shape} on {a_flat.device} but B energy "
                             f"{b_flat.shape} on {b_flat.device}")
        size = a_flat.shape[0]
        side = round(size ** (1.0 / 3.0))
        if side ** 3 != size:
            raise ValueError(f"{size} boxes is not a cube")
        counts, positions, colours = self._buffers_for(size, a_flat.device)
        counts.zero_()
        reference_a, reference_b = (float(value) for value in reference)
        wp.launch(_gather, dim=size, device=a_flat.device,
                  inputs=[a_flat, b_flat, side, reference_a, reference_b,
                          SHOWN * reference_a, SHOWN * reference_b, counts, positions, colours])
        shown, broken, negative = (int(value) for value in counts.numpy())
        if broken:
            raise ValueError(f"{broken} boxes hold an energy that is not finite")
        if shown == 0:
            return Gathered(np.zeros((0, 3), np.float32), np.zeros((0, 4), np.uint8), negative)
        positions_host = positions[:shown].numpy().reshape(shown, 3)
        colours_host = colours[:shown].numpy()
        if a_flat.device.is_cpu:
            # .numpy() of a CPU array shares the reused buffer, so the next gather would
            # overwrite a snapshot still being drawn; take an own copy.
            positions_host, colours_host = positions_host.copy(), colours_host.copy()
        return Gathered(positions_host, colours_host, negative)

    def _buffers_for(self, size: int, device) -> tuple:
        if self._buffers is None or self._buffers[0] != (size, str(device)):
            self._buffers = ((size, str(device)),
                             wp.zeros(3, dtype=wp.int32, device=device),
                             wp.empty(size, dtype=wp.vec3, device=device),
                             wp.empty((size, 4), dtype=wp.uint8, device=device))
        return self._buffers[1:]


def _flat(array: wp.array) -> wp.array:
    return array if array.ndim == 1 else array.flatten()
