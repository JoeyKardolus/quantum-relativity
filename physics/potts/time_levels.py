"""Time levels: each cell's own substep this step, and the boxes ordered by it (local clocks).

Contract: `time_levels(universe, settings)` returns M, the finest level any cell needs,
and prepares the fluid substeps of one step (step.py), on the GPU:
  cell_speed  per cell the largest (|u_x| + |u_y| + |u_z| + sound) * clock over its
              boxes holding more than VAPOUR (a tenth) of its mean energy per box (a box
              just claimed and filling is left out: flow.py lets any box send at most
              a sixth of what it holds per face, so it cannot go negative)
  cell_level  the smallest m >= 0 with cell_speed * step_time / 2**m <= courant: the
              cell advances in substeps of step_time / 2**m of the counter, each lasting
              clock times that in its own time (bible: time is local). A packed cell's
              slow clock gives it a low level: fewer, longer updates.
  box_level   every occupied box gets its cell's level (-1 for an empty box)
  and in universe.scratch: "kick_order" the occupied boxes (flat indices) ordered by
  level from M down, "flow_order" the same boxes ordered by their flow level (the
  finest level among the box and its face neighbours: a box must take in what a finer
  neighbour sends), and "kick_prefix", "flow_prefix" (host lists, index L) the number
  of boxes at level >= L, so the boxes due at a substep are a prefix of each order.
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
from physics.potts.universe import QUANTUM, Universe

_QUANTUM = wp.constant(QUANTUM)
VAPOUR = 0.1
LEVELS = 32


@wp.kernel
def _speeds(owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
            momentum: wp.array3d(dtype=wp.vec3), clock: wp.array(dtype=wp.float32),
            cell_energy: wp.array(dtype=wp.int64), cell_boxes: wp.array(dtype=wp.int32), sound: float,
            vapour: float, low: wp.vec3i, cell_speed: wp.array(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    i = ti + low[0]
    j = tj + low[1]
    k = tk + low[2]
    cell = owner[i, j, k]
    if cell < 0:
        return
    held = float(energy[i, j, k]) * _QUANTUM
    mean = float(cell_energy[cell]) * _QUANTUM / float(wp.max(cell_boxes[cell], 1))
    if held <= vapour * mean or held <= 0.0:
        return
    flow = momentum[i, j, k] / held
    speed = (wp.abs(flow[0]) + wp.abs(flow[1]) + wp.abs(flow[2]) + sound) * clock[cell]
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


@wp.func
def _flow_level(owner: wp.array3d(dtype=wp.int32), box_level: wp.array3d(dtype=wp.int32), here: wp.vec3i,
                side: int) -> int:
    level = box_level[here[0], here[1], here[2]]
    for face in range(6):
        there = here + direction(face)
        if box_owner(owner, there[0], there[1], there[2], side) >= 0:
            level = wp.max(level, box_level[there[0], there[1], there[2]])
    return level


@wp.kernel
def _flow_counts(owner: wp.array3d(dtype=wp.int32), box_level: wp.array3d(dtype=wp.int32), low: wp.vec3i,
                 side: int, counts: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    here = wp.vec3i(ti + low[0], tj + low[1], tk + low[2])
    if owner[here[0], here[1], here[2]] < 0:
        return
    wp.atomic_add(counts, _flow_level(owner, box_level, here, side), 1)


@wp.kernel
def _place(owner: wp.array3d(dtype=wp.int32), box_level: wp.array3d(dtype=wp.int32), low: wp.vec3i, side: int,
           kick_start: wp.array(dtype=wp.int32), flow_start: wp.array(dtype=wp.int32),
           kick_cursor: wp.array(dtype=wp.int32), flow_cursor: wp.array(dtype=wp.int32),
           kick_order: wp.array(dtype=wp.int32), flow_order: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    here = wp.vec3i(ti + low[0], tj + low[1], tk + low[2])
    if owner[here[0], here[1], here[2]] < 0:
        return
    index = (here[0] * side + here[1]) * side + here[2]
    own = box_level[here[0], here[1], here[2]]
    kick_order[kick_start[own] + wp.atomic_add(kick_cursor, own, 1)] = index
    widest = _flow_level(owner, box_level, here, side)
    flow_order[flow_start[widest] + wp.atomic_add(flow_cursor, widest, 1)] = index


def time_levels(universe: Universe, settings: Settings) -> int:
    """Set every cell's time level and order the boxes by level; return the finest level M."""
    device = universe.owner.device
    low = wp.vec3i(*universe.low)
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    universe.cell_speed.zero_()
    wp.launch(_speeds, dim=dims, device=device,
              inputs=[universe.owner, universe.energy, universe.momentum, universe.cell_clock, universe.cell_energy,
                      universe.cell_boxes, settings.sound, VAPOUR, low, universe.cell_speed])
    finest = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(_cell_levels, dim=universe.cell_count, device=device,
              inputs=[universe.cell_speed, settings.step_time, settings.courant, universe.cell_level, finest])
    top = int(finest.numpy()[0])
    if top >= 999:
        raise ValueError(f"step {universe.steps + 1}: a flow speed is not finite")
    if top > settings.max_level:
        raise ValueError(f"step {universe.steps + 1}: a cell needs time level {top}, over {int(settings.max_level)}")
    kick_counts = wp.zeros(LEVELS, dtype=wp.int32, device=device)
    flow_counts = wp.zeros(LEVELS, dtype=wp.int32, device=device)
    wp.launch(_box_levels, dim=dims, device=device,
              inputs=[universe.owner, universe.cell_level, low, universe.box_level, kick_counts])
    wp.launch(_flow_counts, dim=dims, device=device,
              inputs=[universe.owner, universe.box_level, low, universe.side, flow_counts])
    kick_host = kick_counts.numpy()
    flow_host = flow_counts.numpy()
    occupied = int(kick_host.sum())
    order_size = universe.scratch.get("order_size", 0)
    if occupied > order_size:
        order_size = int(occupied * 1.5) + 1024
        universe.scratch["order_size"] = order_size
        universe.scratch["kick_order"] = wp.zeros(order_size, dtype=wp.int32, device=device)
        universe.scratch["flow_order"] = wp.zeros(order_size, dtype=wp.int32, device=device)
    kick_start, kick_prefix = _starts(kick_host)
    flow_start, flow_prefix = _starts(flow_host)
    wp.launch(_place, dim=dims, device=device,
              inputs=[universe.owner, universe.box_level, low, universe.side,
                      wp.array(kick_start, dtype=wp.int32, device=device),
                      wp.array(flow_start, dtype=wp.int32, device=device),
                      wp.zeros(LEVELS, dtype=wp.int32, device=device), wp.zeros(LEVELS, dtype=wp.int32, device=device),
                      universe.scratch["kick_order"], universe.scratch["flow_order"]])
    universe.scratch["kick_prefix"] = kick_prefix
    universe.scratch["flow_prefix"] = flow_prefix
    universe.scratch["kick_counts"] = kick_host
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

