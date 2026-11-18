"""Press joints: how hard the pull presses each joint, against how hard the joint pushes back.

Contract: `press_joints(state, settings)` reads `pull`, `cell_quanta`, `length`, `kind`,
the way table and the neighbour tables, and writes `way_press`, `press`, `squeeze` and
`squeezed_way`. Nothing else.
  way_press[slot]      of every living way, by press_law.py. For each of its two volumes
                       the far set is the volume's neighbours on the side away from the
                       joint (their code, seen from the volume, is the opposite of its
                       mouth); the far pull and far length are the plain means of those that
                       hold energy (an empty cell is empty space and is not read); with
                       no such neighbour the far pull is the volume's own and the far
                       length 0. Only the one pull field is read. Each volume's
                       neighbours are summed per direction once (one thread a volume),
                       then every way reads its two sums: a merged volume with
                       thousands of joints is not walked again for each of its ways.
  press[joint]         its largest `way_press`, never a sum over its ways
  squeezed_way[joint]  the slot of that way, a tie to the lowest slot
  squeeze[joint]       `max(press, 0)` over the joint's own push (push_law.py: only
                       minus pushes, asked of sign_law.py)
A living joint with no way has press 0, squeeze 0 and `squeezed_way` -1. An id that is
not a joint has 0, 0 and -1 too.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_2_swirls.definitions.sign_law import makes_pull, pushes
from physics.engine.axiom_3_structure.definitions.direction_codes import opposite
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, NO_CELL
from physics.engine.axiom_3_structure.definitions.neighbour_of import neighbour_code, neighbour_count, neighbour_id
from physics.engine.axiom_5_packing.definitions.press_law import way_press_law
from physics.engine.axiom_5_packing.definitions.push_law import push_law, squeeze_law

wp.set_module_options({"fuse_fp": False})

_NO_WAY_YET = wp.constant(wp.int32(2 ** 31 - 1))
_ENERGY_VOLUME = wp.constant(wp.int32(ENERGY_VOLUME))
_NO_CELL = wp.constant(wp.int32(NO_CELL))
_DIRECTIONS = 27


@wp.kernel
def _far_sides(kind: wp.array(dtype=wp.int32), adj_at: wp.array(dtype=wp.int32), adj_n: wp.array(dtype=wp.int32),
               adj_id: wp.array(dtype=wp.int32), adj_code: wp.array(dtype=wp.int32),
               pull: wp.array(dtype=wp.float64), length: wp.array(dtype=wp.float32),
               cell_quanta: wp.array(dtype=wp.int64), pull_sum: wp.array2d(dtype=wp.float64), length_sum: wp.array2d(dtype=wp.float64),
               number: wp.array2d(dtype=wp.int32)):
    """Per energy volume and direction: the summed pull and length of its neighbours there, and how many."""
    cell = wp.tid()
    if not makes_pull(kind[cell]):
        return
    for index in range(neighbour_count(adj_n, cell)):
        code = neighbour_code(adj_at, adj_code, cell, index)
        other = neighbour_id(adj_at, adj_id, cell, index)
        if kind[other] == _NO_CELL or cell_quanta[other] <= wp.int64(0):
            continue
        pull_sum[cell, code] = pull_sum[cell, code] + pull[other]
        length_sum[cell, code] = length_sum[cell, code] + wp.float64(length[other])
        number[cell, code] = number[cell, code] + 1


@wp.func
def _far_and_reach(volume: wp.int32, mouth: wp.int32, joint_length: wp.float64,
                   pull_sum: wp.array2d(dtype=wp.float64), length_sum: wp.array2d(dtype=wp.float64),
                   number: wp.array2d(dtype=wp.int32), pull: wp.array(dtype=wp.float64),
                   length: wp.array(dtype=wp.float32)) -> wp.vec2d:
    """The pull behind one volume of a way, and the reach from the joint's middle to there."""
    away = opposite(mouth)
    behind = number[volume, away]
    far = pull[volume]
    far_length = wp.float64(0.0)
    if behind > 0:
        far = pull_sum[volume, away] / wp.float64(behind)
        far_length = length_sum[volume, away] / wp.float64(behind)
    reach = joint_length / wp.float64(2.0) + wp.float64(length[volume]) + far_length / wp.float64(2.0)
    return wp.vec2d(far, reach)


@wp.kernel
def _open_joints(press: wp.array(dtype=wp.float32), squeezed_way: wp.array(dtype=wp.int32)):
    cell = wp.tid()
    press[cell] = wp.float32(-wp.inf)
    squeezed_way[cell] = _NO_WAY_YET


@wp.kernel
def _press_ways(way_joint: wp.array(dtype=wp.int32), way_a: wp.array(dtype=wp.int32),
                way_b: wp.array(dtype=wp.int32), way_mouth_a: wp.array(dtype=wp.int32),
                way_mouth_b: wp.array(dtype=wp.int32), pull_sum: wp.array2d(dtype=wp.float64),
                length_sum: wp.array2d(dtype=wp.float64), number: wp.array2d(dtype=wp.int32),
                pull: wp.array(dtype=wp.float64),
                length: wp.array(dtype=wp.float32), cell_quanta: wp.array(dtype=wp.int64),
                way_press: wp.array(dtype=wp.float32), press: wp.array(dtype=wp.float32)):
    slot = wp.tid()
    joint = way_joint[slot]
    if joint < 0:
        return
    joint_length = wp.float64(length[joint])
    side_a = _far_and_reach(way_a[slot], way_mouth_a[slot], joint_length, pull_sum, length_sum, number, pull,
                            length)
    side_b = _far_and_reach(way_b[slot], way_mouth_b[slot], joint_length, pull_sum, length_sum, number, pull,
                            length)
    pressed = wp.float32(way_press_law(wp.float64(cell_quanta[way_a[slot]]), side_a[0], side_a[1],
                                       wp.float64(cell_quanta[way_b[slot]]), side_b[0], side_b[1],
                                       pull[joint]))
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
    pull_sum = wp.zeros((state.count, _DIRECTIONS), dtype=wp.float64, device=device)
    length_sum = wp.zeros((state.count, _DIRECTIONS), dtype=wp.float64, device=device)
    number = wp.zeros((state.count, _DIRECTIONS), dtype=wp.int32, device=device)
    wp.launch(_far_sides, dim=state.count,
              inputs=[state.kind, state.adj_at, state.adj_n, state.adj_id, state.adj_code, state.pull,
                      state.length, state.cell_quanta],
              outputs=[pull_sum, length_sum, number], device=device)
    wp.launch(_press_ways, dim=state.way_count,
              inputs=[state.way_joint, state.way_a, state.way_b, state.way_mouth_a, state.way_mouth_b,
                      pull_sum, length_sum, number, state.pull, state.length, state.cell_quanta],
              outputs=[state.way_press, state.press], device=device)
    wp.launch(_pick_hardest, dim=state.way_count, inputs=[state.way_joint, state.way_press, state.press],
              outputs=[state.squeezed_way], device=device)
    wp.launch(_squeeze_joints, dim=state.count,
              inputs=[state.kind, state.cell_quanta, state.length, wp.float64(settings.push_strength)],
              outputs=[state.press, state.squeezed_way, state.squeeze], device=device)
