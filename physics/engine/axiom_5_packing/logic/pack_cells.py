"""Pack cells: how packed each cell is, and how slow its clock runs for it.

Contract: `pack_cells(state, settings)` reads `kind`, `cell_quanta` and `length` and
writes, for every id handed out, `packing` (packing_law.py: quanta over length cubed)
and `rate` (clock_rate.py: `1 / (1 + packing / settings.clock_packing)`). One law for
energy volumes and joints. An id that is no cell, and an empty cell, have packing 0 and
rate 1. Nothing else is written. The step calls it after every `size_cells`, so packing
and rate always belong to the quanta the cell holds now.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL
from physics.engine.axiom_5_packing.definitions.clock_rate import clock_rate
from physics.engine.axiom_5_packing.definitions.packing_law import packing_law

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(wp.int32(NO_CELL))


@wp.kernel
def _pack(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
          length: wp.array(dtype=wp.float32), clock_packing: wp.float64,
          packing: wp.array(dtype=wp.float64), rate: wp.array(dtype=wp.float32)):
    cell = wp.tid()
    packed = wp.float64(0.0)
    if kind[cell] != _NO_CELL:
        packed = packing_law(cell_quanta[cell], wp.float64(length[cell]))
    packing[cell] = packed
    rate[cell] = clock_rate(packed, clock_packing)


def pack_cells(state, settings) -> None:
    """Write every cell's packing and clock rate from its quanta and length."""
    wp.launch(_pack, dim=state.count,
              inputs=[state.kind, state.cell_quanta, state.length, wp.float64(settings.clock_packing)],
              outputs=[state.packing, state.rate], device=state.device)
