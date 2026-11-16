"""Later substeps: the fluid substeps after the first, run only over the boxes of the cells that need them.

Contract: `later_substeps(state, settings, duration)` runs, for the boxes substeps.py
listed (state.later: the boxes of every cell needing more than one fluid substep, cells
of the highest count first), substeps 1 .. count - 1: each the same Strang splitting
step.py runs over all boxes (site_turn for `duration` / count, the hops along x, y, z
with parity 0 then 1, the hops in reverse order, site_turn again; hop_pair.py,
site_turn.py), launched over the listed boxes whose cell still steps, with the same hash
seeds (pass_seed.py) and each cell's own count. The fluid never leaves its cell, so the
result is, to the last bit, what launches over all boxes would give with the same
counts; but a universe where one cell in a hundred is steep no longer pays that cell's
substeps for every box (numerics, 2026-11-17). Does nothing when nothing is listed.
Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.hop_pair import hop_pair
from physics.engine.pass_seed import pass_seed
from physics.engine.settings import Settings
from physics.engine.site_turn import site_turn
from physics.engine.state import State

wp.set_module_options({"fuse_fp": False})

_ORDER = [(axis, parity) for axis in range(3) for parity in range(2)]


@wp.func
def _box(index: int, side: int) -> wp.vec3i:
    """The stacked arrays' three indices of listed box `index`."""
    return wp.vec3i(index // (side * side), (index // side) % side, index % side)


@wp.kernel
def _turn(boxes: wp.array(dtype=wp.int32), quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32),
          pull: wp.array3d(dtype=wp.float32), phase: wp.array3d(dtype=wp.float32), duration: float,
          rates: wp.array2d(dtype=wp.float32), pushes: wp.array2d(dtype=wp.float32),
          packings: wp.array2d(dtype=wp.float32), pressure: float, side: int, plans: wp.array2d(dtype=wp.int32),
          sub_index: int):
    here = _box(boxes[wp.tid()], side)
    amount = quanta[here[0], here[1], here[2]]
    if amount == 0:
        return
    home = here[0] // side
    cell = owner[here[0], here[1], here[2]]
    count = wp.abs(plans[home, cell])
    if sub_index >= count:
        return
    span = duration / float(count)
    phase[here[0], here[1], here[2]] = site_turn(amount, pull[here[0], here[1], here[2]], pushes[home, cell],
                                                 packings[home, cell], pressure, phase[here[0], here[1], here[2]],
                                                 span, rates[home, cell])


@wp.kernel
def _hop(boxes: wp.array(dtype=wp.int32), quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32),
         owner: wp.array3d(dtype=wp.int32), rates: wp.array2d(dtype=wp.float32), side: int, wrap: int, axis: int,
         parity: int, duration: float, seed: int, keys: wp.array(dtype=wp.int32),
         plans: wp.array2d(dtype=wp.int32), sub_index: int):
    here = _box(boxes[wp.tid()], side)
    home = here[0] // side
    local = wp.vec3i(here[0] - home * side, here[1], here[2])
    if local[axis] % 2 != parity:
        return
    beyond = local[axis] + 1
    if beyond >= side:
        if wrap == 0:
            return
        beyond = 0
    there = here
    if axis == 0:
        there[0] = home * side + beyond
    else:
        there[axis] = beyond
    if quanta[here[0], here[1], here[2]] == 0 or quanta[there[0], there[1], there[2]] == 0:
        return
    cell = owner[here[0], here[1], here[2]]
    if owner[there[0], there[1], there[2]] != cell:
        return
    count = wp.abs(plans[home, cell])
    if sub_index >= count:
        return
    tau = duration / float(count) * rates[home, cell]
    hop_pair(quanta, phase, here, there, tau, seed ^ keys[home], (local[0] * side + local[1]) * side + local[2])


def later_substeps(state: State, settings: Settings, duration: float) -> None:
    """Run substeps 1 .. count - 1 over the listed boxes whose cells still step."""
    later = state.later
    if later is None or later.get("boxes") is None:
        return
    for sub_index in range(1, len(later["ends"])):
        _substep(state, settings, duration, sub_index, int(later["ends"][sub_index]))


def _substep(state: State, settings: Settings, duration: float, sub_index: int, count: int) -> None:
    """One Strang substep over the first `count` listed boxes."""
    device = state.quanta.device
    boxes = state.later["boxes"]
    turn = [boxes, state.quanta, state.owner, state.pull, state.phase, float(duration), state.cells_d["rate"],
            state.cells_d["push"], state.cells_d["packing"], float(settings.push), state.side, state.substeps,
            int(sub_index)]
    wp.launch(_turn, dim=count, device=device, inputs=turn)
    passes = list(enumerate(_ORDER)) + [(6 + index, pair) for index, pair in enumerate(_ORDER[::-1])]
    for pass_index, (axis, parity) in passes:
        wp.launch(_hop, dim=count, device=device,
                  inputs=[boxes, state.quanta, state.phase, state.owner, state.cells_d["rate"], state.side,
                          int(state.wrap), int(axis), int(parity), float(duration),
                          int(pass_seed(state.step, sub_index, pass_index)), state.keys, state.substeps,
                          int(sub_index)])
    wp.launch(_turn, dim=count, device=device, inputs=turn)
