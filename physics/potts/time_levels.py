"""Time levels: each cell's own substep this step, and the boxes ordered by it (local clocks).

Contract: `time_levels(universe, settings)` returns M, the finest level any cell needs,
and prepares the fluid substeps of one step (step.py), on the GPU:
  cell_speed  per cell the largest (|u_x| + |u_y| + |u_z| + s) * clock over its boxes
              holding more than VAPOUR (a tenth) of its mean energy per box, u the
              energy fluid's flow (the phase difference to the box's face neighbours
              in its own cell, averaged per axis, hbar = box mass = 1) and s the local
              sound speed sqrt(g |psi|**2), g = sound**2 / rest (on_site.py; a joint's
              push term is left out of s)
  cell_level  the smallest m >= 0 with cell_speed * step_time / 2**m <= courant: the
              cell advances in substeps of step_time / 2**m of the counter, each lasting
              clock times that in its own time (bible: time is local). A packed cell's
              slow clock gives it a low level: fewer, longer updates. The hops and turns
              are exact rotations at any dt (hop.py, on_site.py); the level keeps them
              accurate, not stable.
  box_level   every occupied box gets its cell's level (-1 for an empty box)
  and in universe.scratch: "site_order" the occupied boxes (flat indices) ordered by
  level from M down, "site_prefix" (host list, index L) the number of boxes at level
  >= L, so the boxes due at a substep are a prefix, and "site_counts" per level.
With M levels a step has 2**M finest substeps; at substep s the levels
>= M - (trailing zeros of s) are due (all at s = 0). Raises ValueError when a speed is
not finite or a cell needs a level above settings.max_level (runaway guard).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.potts.box_owner import box_owner
from physics.potts.direction import direction
from physics.potts.settings import Settings
from physics.potts.universe import Universe
VAPOUR = 0.1
LEVELS = 32


@wp.func
def _phase_step(psi: wp.array3d(dtype=wp.vec2d), owner: wp.array3d(dtype=wp.int32), here: wp.vec3i, face: int,
                cell: int, side: int) -> float:
    """The phase turn from `here` to its face neighbour, 0 when that box is not in the same cell."""
    there = here + direction(face)
    if box_owner(owner, there[0], there[1], there[2], side) != cell:
        return 0.0
    first = psi[here[0], here[1], here[2]]
    second = psi[there[0], there[1], there[2]]
    return float(wp.atan2(first[0] * second[1] - first[1] * second[0], first[0] * second[0] + first[1] * second[1]))


@wp.kernel
def _speeds(owner: wp.array3d(dtype=wp.int32), psi: wp.array3d(dtype=wp.vec2d), clock: wp.array(dtype=wp.float32),
            cell_energy: wp.array(dtype=wp.float64), cell_boxes: wp.array(dtype=wp.int32), fluid_g: float,
            vapour: float, low: wp.vec3i, side: int, cell_speed: wp.array(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    here = wp.vec3i(ti + low[0], tj + low[1], tk + low[2])
    cell = owner[here[0], here[1], here[2]]
    if cell < 0:
        return
    held = float(wp.dot(psi[here[0], here[1], here[2]], psi[here[0], here[1], here[2]]))
    mean = float(cell_energy[cell]) / float(wp.max(cell_boxes[cell], 1))
    if held <= vapour * mean or held <= 0.0:
        return
    speed = wp.sqrt(fluid_g * held)
    for axis in range(3):
        speed += 0.5 * wp.abs(_phase_step(psi, owner, here, 2 * axis, cell, side)
                              - _phase_step(psi, owner, here, 2 * axis + 1, cell, side))
    speed = speed * clock[cell]
    if not wp.isfinite(speed):
        speed = 1.0e30
    wp.atomic_max(cell_speed, cell, speed)


@wp.kernel
def _cell_levels(cell_speed: wp.array(dtype=wp.float32), step_time: float, courant: float,
                 cell_level: wp.array(dtype=wp.int32), finest: wp.array(dtype=wp.int32)):
    cell = wp.tid()
    need = cell_speed[cell] * step_time / courant
    level = int(0)
    if need >= 1.0e29:
        level = 999
    elif need > 1.0:
        level = int(wp.ceil(wp.log2(need)))
        if wp.pow(2.0, float(level)) < need:
            level += 1
    cell_level[cell] = level
    wp.atomic_max(finest, 0, level)


@wp.kernel
def _box_levels(owner: wp.array3d(dtype=wp.int32), cell_level: wp.array(dtype=wp.int32), low: wp.vec3i,
                box_level: wp.array3d(dtype=wp.int32), counts: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    i = ti + low[0]
    j = tj + low[1]
    k = tk + low[2]
    cell = owner[i, j, k]
    level = int(-1)
    if cell >= 0:
        level = cell_level[cell]
        wp.atomic_add(counts, level, 1)
    box_level[i, j, k] = level


@wp.kernel
def _place(owner: wp.array3d(dtype=wp.int32), box_level: wp.array3d(dtype=wp.int32), low: wp.vec3i, side: int,
           site_start: wp.array(dtype=wp.int32), site_cursor: wp.array(dtype=wp.int32),
           site_order: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    here = wp.vec3i(ti + low[0], tj + low[1], tk + low[2])
    if owner[here[0], here[1], here[2]] < 0:
        return
    index = (here[0] * side + here[1]) * side + here[2]
    own = box_level[here[0], here[1], here[2]]
    site_order[site_start[own] + wp.atomic_add(site_cursor, own, 1)] = index


def time_levels(universe: Universe, settings: Settings) -> int:
    """Set every cell's time level and order the boxes by level; return the finest level M."""
    device = universe.owner.device
    low = wp.vec3i(*universe.low)
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    universe.cell_speed.zero_()
    wp.launch(_speeds, dim=dims, device=device,
              inputs=[universe.owner, universe.psi, universe.cell_clock, universe.cell_energy, universe.cell_boxes,
                      settings.sound * settings.sound / universe.rest, VAPOUR, low, universe.side,
                      universe.cell_speed])
    finest = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(_cell_levels, dim=universe.cell_count, device=device,
              inputs=[universe.cell_speed, settings.step_time, settings.courant, universe.cell_level, finest])
    top = int(finest.numpy()[0])
    if top >= 999:
        raise ValueError(f"step {universe.steps + 1}: a flow speed is not finite")
    if top > settings.max_level:
        raise ValueError(f"step {universe.steps + 1}: a cell needs time level {top}, over {int(settings.max_level)}")
    site_counts = wp.zeros(LEVELS, dtype=wp.int32, device=device)
    wp.launch(_box_levels, dim=dims, device=device,
              inputs=[universe.owner, universe.cell_level, low, universe.box_level, site_counts])
    site_host = site_counts.numpy()
    occupied = int(site_host.sum())
    order_size = universe.scratch.get("order_size", 0)
    if occupied > order_size:
        order_size = int(occupied * 1.5) + 1024
        universe.scratch["order_size"] = order_size
        universe.scratch["site_order"] = wp.zeros(order_size, dtype=wp.int32, device=device)
    site_start, site_prefix = _starts(site_host)
    wp.launch(_place, dim=dims, device=device,
              inputs=[universe.owner, universe.box_level, low, universe.side,
                      wp.array(site_start, dtype=wp.int32, device=device),
                      wp.zeros(LEVELS, dtype=wp.int32, device=device), universe.scratch["site_order"]])
    universe.scratch["site_prefix"] = site_prefix
    universe.scratch["site_counts"] = site_host
    return top


def _starts(counts: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Where each level's run starts in an order running from the finest level down, and the prefix sizes."""
    starts = np.zeros(LEVELS, dtype=np.int32)
    running = 0
    for level in range(LEVELS - 1, -1, -1):
        starts[level] = running
        running += int(counts[level])
    prefix = [int(counts[level:].sum()) for level in range(LEVELS)]
    return starts, prefix

