"""Press joints: how hard the pull presses each joint, against how hard the joint pushes back.

Contract: `press_joints(state, settings)` reads `pull`, `cell_quanta`, `length`, `kind`
and the way table, and writes `way_press`, `press`, `squeeze` and `squeezed_way`.
Nothing else.
  way_press[slot]      of every living way, by press_law.py: the slope of the pull from
                       the joint down into each of its two volumes, the reach of a side
                       being half the joint's length plus half the volume's. Only the
                       one pull field is read, at the joint and at its two volumes.
  press[joint]         its largest `way_press`, never a sum over its ways
  squeezed_way[joint]  the slot of that way, a tie to the lowest slot
  squeeze[joint]       `max(press, 0)` over the joint's own push (push_law.py: only
                       minus pushes, asked of sign_law.py)
A living joint with no way has press 0, squeeze 0 and `squeezed_way` -1. An id that is
not a joint has 0, 0 and -1 too.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_2_swirls.definitions.sign_law import pushes
from physics.engine.axiom_5_packing.definitions.press_law import way_press_law
from physics.engine.axiom_5_packing.definitions.push_law import push_law, squeeze_law

wp.set_module_options({"fuse_fp": False})

_NO_WAY_YET = wp.constant(wp.int32(2 ** 31 - 1))
_HALF = wp.constant(wp.float64(0.5))


@wp.kernel
def _open_joints(press: wp.array(dtype=wp.float32), squeezed_way: wp.array(dtype=wp.int32)):
    cell = wp.tid()
    press[cell] = wp.float32(-wp.inf)
    squeezed_way[cell] = _NO_WAY_YET


@wp.kernel
def _press_ways(way_joint: wp.array(dtype=wp.int32), way_a: wp.array(dtype=wp.int32),
                way_b: wp.array(dtype=wp.int32), pull: wp.array(dtype=wp.float64),
                length: wp.array(dtype=wp.float32), cell_quanta: wp.array(dtype=wp.int64),
                way_press: wp.array(dtype=wp.float32), press: wp.array(dtype=wp.float32)):
    slot = wp.tid()
    joint = way_joint[slot]
    if joint < 0:
        return
    volume_a = way_a[slot]
    volume_b = way_b[slot]
    joint_length = wp.float64(length[joint])
    reach_a = _HALF * (joint_length + wp.float64(length[volume_a]))
    reach_b = _HALF * (joint_length + wp.float64(length[volume_b]))
    pressed = wp.float32(way_press_law(wp.float64(cell_quanta[volume_a]), pull[volume_a], reach_a,
                                       wp.float64(cell_quanta[volume_b]), pull[volume_b], reach_b, pull[joint]))
    way_press[slot] = pressed
    wp.atomic_max(press, joint, pressed)


@wp.kernel
def _pick_hardest(way_joint: wp.array(dtype=wp.int32), way_press: wp.array(dtype=wp.float32),
                  press: wp.array(dtype=wp.float32), squeezed_way: wp.array(dtype=wp.int32)):
    slot = wp.tid()
    joint = way_joint[slot]
    if joint < 0:
        return
    if way_press[slot] == press[joint]:
        wp.atomic_min(squeezed_way, joint, slot)


@wp.kernel
def _squeeze_joints(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
                    length: wp.array(dtype=wp.float32), push_strength: wp.float64,
                    press: wp.array(dtype=wp.float32), squeezed_way: wp.array(dtype=wp.int32),
                    squeeze: wp.array(dtype=wp.float32)):
    cell = wp.tid()
    if squeezed_way[cell] == _NO_WAY_YET or not pushes(kind[cell]):
        press[cell] = wp.float32(0.0)
        squeeze[cell] = wp.float32(0.0)
        squeezed_way[cell] = -1
        return
    push = push_law(cell_quanta[cell], wp.float64(length[cell]), push_strength)
    squeeze[cell] = squeeze_law(wp.float64(press[cell]), push)


def press_joints(state, settings) -> None:
    """Write every way's press, and every joint's hardest way, press and squeeze."""
    device = state.device
    wp.launch(_open_joints, dim=state.count, outputs=[state.press, state.squeezed_way], device=device)
    wp.launch(_press_ways, dim=state.way_count,
              inputs=[state.way_joint, state.way_a, state.way_b, state.pull, state.length, state.cell_quanta],
              outputs=[state.way_press, state.press], device=device)
    wp.launch(_pick_hardest, dim=state.way_count, inputs=[state.way_joint, state.way_press, state.press],
              outputs=[state.squeezed_way], device=device)
    wp.launch(_squeeze_joints, dim=state.count,
              inputs=[state.kind, state.cell_quanta, state.length, wp.float64(settings.push_strength)],
              outputs=[state.press, state.squeezed_way, state.squeeze], device=device)
