"""Diffusion step: the trivial local step that tests the bench harness, not physics.

Contract: `diffusion_step(batch)` advances every universe by one step in place and adds 1
to batch.steps. Step s pairs each box with its next box along axis s % 3, pairs starting
at even coordinates on even rounds (s // 3 even) and odd ones on odd rounds, so the pairs
never overlap and the kernel has no races (Margolus-style blocks). A pair whose two
boxes hold the same sign (both plus or both minus) shares its amount evenly, the odd
quantum staying with the box that held more, and both take the circular mean of their
phases; a pair of opposite signs, or with a zero (no space), is left alone, so plus and
minus never meet and every universe's plus amount and minus amount are kept exactly
(bible rule 4). Pairs never cross a universe's wall, so universes never mix. The result
does not depend on which universes share the batch.
"""
from __future__ import annotations

import warp as wp

from physics.bench.quanta_batch import QuantaBatch


@wp.kernel
def _share(quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float32), axis: int, offset: int):
    universe, ix, iy, iz = wp.tid()
    side = quanta.shape[1]
    along = ix
    if axis == 1:
        along = iy
    if axis == 2:
        along = iz
    if (along - offset) % 2 != 0 or along + 1 >= side:
        return
    jx = ix
    jy = iy
    jz = iz
    if axis == 0:
        jx = ix + 1
    if axis == 1:
        jy = iy + 1
    if axis == 2:
        jz = iz + 1
    first = quanta[universe, ix, iy, iz]
    second = quanta[universe, jx, jy, jz]
    same_sign = (first > 0 and second > 0) or (first < 0 and second < 0)
    if not same_sign:
        return
    sign = 1
    if first < 0:
        sign = -1
    held_first = wp.abs(first)
    held_second = wp.abs(second)
    amount = held_first + held_second
    half = amount / 2
    if held_first >= held_second:
        quanta[universe, ix, iy, iz] = sign * (amount - half)
        quanta[universe, jx, jy, jz] = sign * half
    else:
        quanta[universe, ix, iy, iz] = sign * half
        quanta[universe, jx, jy, jz] = sign * (amount - half)
    turn_first = phase[universe, ix, iy, iz]
    turn_second = phase[universe, jx, jy, jz]
    mean = wp.atan2(wp.sin(turn_first) + wp.sin(turn_second), wp.cos(turn_first) + wp.cos(turn_second))
    phase[universe, ix, iy, iz] = mean
    phase[universe, jx, jy, jz] = mean


def diffusion_step(batch: QuantaBatch) -> None:
    """One race-free pairwise sharing of same-sign quanta along one axis, every universe at once."""
    axis = batch.steps % 3
    offset = (batch.steps // 3) % 2
    wp.launch(_share, dim=tuple(batch.quanta.shape), inputs=[batch.quanta, batch.phase, axis, offset],
              device=batch.quanta.device)
    batch.steps += 1
