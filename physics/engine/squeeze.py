"""Squeeze: volumes pull on each other across the joint between them, the joint pushes back, and a squeezed joint makes minus energy out of itself.

Contract: `squeeze(state, settings)` works, for every joint that carries a way (way.py:
it lies between two energy volumes), on the GPU:
  pull across  the sum over its ways (a, b) of G * Qa * Qb / D ** 2, with Qa, Qb the two
               volumes' quanta and D the distance between their middles in cell sizes,
               length_a / 2 + the joint's length + length_b / 2
  push         P * (Qj / length_j) ** 2, the joint's own quanta against its own size;
               by the size law that is P * Qj ** 4 / constant ** 2, steep in its energy
  squeeze      pull across / push, kept in state.squeeze; above 1 the joint is squeezed
A squeezed joint makes new minus quanta, out of itself: squeeze_rate * Qj * (squeeze -
1) * step_time * its clock rate, never more than brings its push level with the pull
(Qj * (squeeze ** (1 / 4) - 1)), in whole quanta, floor(amount + uniform) (the uniform a
hash of the joint, the state's key and the step). They are added to its gain (pour.py
puts them in its boxes) and to state.books[0]; nothing is taken from any volume, so the
total energy rises (axiom 1). A joint that is not squeezed makes nothing: its energy
only ever rises, so it only ever thins (axiom 6). A squeezed joint whose quanta, with
what it just made, reach collapse_thin times the start's mean joint (that many times
thinner) is flagged to collapse (state.flags 1, counted in state.found[0]). Joints with
no way (edge and corner joints, joints shut inside one volume) are not squeezed: their
squeeze is 0. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.settings import Settings
from physics.engine.state import BOXES, JOINT, State
from physics.engine.way import way

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _squeeze(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
             length: wp.array(dtype=wp.float32), rate: wp.array(dtype=wp.float32),
             ways_n: wp.array(dtype=wp.int32), way_at: wp.array(dtype=wp.int32),
             way_pool: wp.array(dtype=wp.int32), side_n: wp.array(dtype=wp.int32),
             side: int, pull_strength: wp.float64, push_strength: wp.float64,
             make_rate: wp.float64, span: wp.float64, limit: wp.float64, seed: int,
             squeezes: wp.array(dtype=wp.float32), gain: wp.array(dtype=wp.int64),
             flags: wp.array(dtype=wp.int32), found: wp.array(dtype=wp.int32), books: wp.array(dtype=wp.int64)):
    joint = wp.tid()
    if kind[joint] != -1:
        return
    flags[joint] = 0
    squeezes[joint] = 0.0
    count = ways_n[joint]
    if count == 0:
        return
    held = wp.float64(cell_quanta[joint])
    own = wp.float64(length[joint])
    across = wp.float64(0.0)
    for index in range(count):
        ends = way(joint, index, side, way_at, side_n, ways_n, way_pool)
        if ends[0] == ends[1]:
            continue
        reach = wp.float64(0.5) * wp.float64(length[ends[0]]) + own + wp.float64(0.5) * wp.float64(length[ends[1]])
        across += pull_strength * wp.float64(cell_quanta[ends[0]]) * wp.float64(cell_quanta[ends[1]]) / (reach * reach)
    push = push_strength * held * held / (own * own)
    pressed = across / push
    squeezes[joint] = wp.float32(pressed)
    if pressed <= wp.float64(1.0):
        return
    wanted = make_rate * held * (pressed - wp.float64(1.0)) * span * wp.float64(rate[joint])
    level = held * (wp.pow(pressed, wp.float64(0.25)) - wp.float64(1.0))
    room = wp.float64(BOXES) * wp.float64(536870912.0) - held
    rng = wp.rand_init(seed, joint)
    made = wp.int64(wp.floor(wp.min(wp.min(wanted, level), room) + wp.float64(wp.randf(rng))))
    if made > wp.int64(0):
        wp.atomic_add(gain, joint, made)
        wp.atomic_add(books, 0, made)
    if held + wp.float64(made) >= limit:
        flags[joint] = 1
        wp.atomic_add(found, 0, 1)


def squeeze(state: State, settings: Settings) -> None:
    """Measure every joint's squeeze, let the squeezed ones make minus quanta, flag those to collapse."""
    seed = (state.step * 64 + 60) ^ state.key
    wp.launch(_squeeze, dim=state.count, device=state.device,
              inputs=[state.kind, state.cell_quanta, state.length, state.rate, state.ways_n, state.way_at,
                      state.way_pool, state.side_n, state.side, wp.float64(settings.pull_strength),
                      wp.float64(settings.push_strength), wp.float64(settings.squeeze_rate),
                      wp.float64(settings.step_time), wp.float64(settings.collapse_thin * state.start_quanta[JOINT]),
                      int(seed & 0x7FFFFFFF), state.squeeze, state.gain, state.flags, state.found, state.books])
