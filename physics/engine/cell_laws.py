"""Cell laws: every cell's size, packing and clock rate from its own quanta, and the volumes grown big enough to tear.

Contract: `cell_laws(state, settings)` sets, for every living cell, on the GPU:
  length   float32(size_energy[kind] / quanta), worked in float64: the size law, more
           energy smaller, for energy volumes and joints alike (axiom 3); nothing else
           ever sets a cell's size
  packing  (start_length[kind] / length) ** 3: how many times less room than the
           start's mean cell of its kind
  rate     its local clock rate, clock_rate(packing, clock_packing): more packed, slower
and flags (value 2, state.flags) every energy volume longer than settings.tear_length
that holds at least 3 quanta per box, counting them in state.found[1]; flags of joints
(squeeze.py) are left. Ids that are not a cell get length 0 and rate 0. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.clock_rate import clock_rate
from physics.engine.settings import Settings
from physics.engine.state import BOXES, ENERGY_VOLUME, JOINT, State


@wp.kernel
def _laws(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64), length: wp.array(dtype=wp.float32),
          packing: wp.array(dtype=wp.float32), rate: wp.array(dtype=wp.float32), flags: wp.array(dtype=wp.int32),
          found: wp.array(dtype=wp.int32), volume_energy: wp.float64, joint_energy: wp.float64, volume_start: float,
          joint_start: float, clock_packing: float, tear_length: float, least: wp.int64):
    cell = wp.tid()
    sort = kind[cell]
    if sort == 0:
        length[cell] = 0.0
        packing[cell] = 0.0
        rate[cell] = 0.0
        return
    held = cell_quanta[cell]
    constant = joint_energy
    start = joint_start
    if sort == 1:
        constant = volume_energy
        start = volume_start
    size = wp.float32(constant / wp.float64(held))
    ratio = start / size
    packed = ratio * ratio * ratio
    length[cell] = size
    packing[cell] = packed
    rate[cell] = clock_rate(packed, clock_packing)
    if sort == 1:
        flags[cell] = 0
        if tear_length > 0.0 and size > tear_length and held >= least:
            flags[cell] = 2
            wp.atomic_add(found, 1, 1)


def cell_laws(state: State, settings: Settings) -> None:
    """Set every cell's length, packing and clock rate from its quanta; flag the volumes to tear."""
    wp.launch(_laws, dim=state.count, device=state.device,
              inputs=[state.kind, state.cell_quanta, state.length, state.packing, state.rate, state.flags,
                      state.found, wp.float64(state.size_energy[ENERGY_VOLUME]),
                      wp.float64(state.size_energy[JOINT]), float(state.start_length[ENERGY_VOLUME]),
                      float(state.start_length[JOINT]), float(settings.clock_packing), float(settings.tear_length),
                      wp.int64(3 * BOXES)])
