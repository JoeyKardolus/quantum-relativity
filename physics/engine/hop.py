"""Hop: the quantum fluid moves whole quanta between face-neighbouring boxes of one cell's own block, one exact pair rotation each.

Contract: `hop(state, settings, axis, parity, duration, pass_index)` advances the
fluid's kinetic step (Schrodinger, hbar = box mass = 1: i dpsi/dt = -1/2 sum over faces
(psi_there - psi) + on-site terms, on_site.py) across every face along `axis` inside
every living cell's block whose lower box has that coordinate's `parity`. A block has
closed walls: no pair reaches out of its cell, so the flow inside a cell is frictionless
and no box's energy leaves by flow (between cells energy moves only by crossing a joint,
cross.py). The scheme is De Raedt's product formula (1987): one exact pair rotation
(pair_rotation.py, hop_pair.py) with turn -tau, tau = duration * the cell's clock rate,
handing over whole quanta; the rounding's uniform comes from a hash of the box's place,
the state's key, the step and the pass. No box drops below 1 quantum and no box changes
sign. Each cell's amount is kept exactly. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.hop_pair import hop_pair
from physics.engine.pass_seed import pass_seed
from physics.engine.settings import Settings
from physics.engine.state import BLOCK, BOXES, State

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _hop(kind: wp.array(dtype=wp.int32), rate: wp.array(dtype=wp.float32), quanta: wp.array4d(dtype=wp.int32),
         phase: wp.array4d(dtype=wp.float32), axis: int, parity: int, duration: float, seed: int, across: int,
         boxes: int):
    cell, one, two, pair = wp.tid()
    if kind[cell] == 0:
        return
    along = parity + 2 * pair
    here = wp.vec3i(along, one, two)
    if axis == 1:
        here = wp.vec3i(one, along, two)
    elif axis == 2:
        here = wp.vec3i(one, two, along)
    there = here
    there[axis] = along + 1
    box_index = cell * boxes + (here[0] * across + here[1]) * across + here[2]
    hop_pair(quanta, phase, cell, here, there, duration * rate[cell], seed, box_index)


def hop(state: State, settings: Settings, axis: int, parity: int, duration: float, pass_index: int) -> None:
    """Move whole quanta across the faces inside each cell's block along `axis` whose lower box has `parity`."""
    pairs = (BLOCK - parity) // 2
    seed = pass_seed(state.step, 0, pass_index) ^ state.key
    wp.launch(_hop, dim=(state.count, BLOCK, BLOCK, pairs), device=state.device,
              inputs=[state.kind, state.rate, state.quanta, state.phase, int(axis), int(parity), float(duration),
                      int(seed & 0x7FFFFFFF), BLOCK, BOXES])
