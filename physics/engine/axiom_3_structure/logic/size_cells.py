"""Size cells: every cell's length from its energy alone.

Contract: `size_cells(state, settings)` writes `length` of every id handed out so far:
`size_law(cell_quanta, size_constant)` for a living cell, energy volume and joint by the
one law and the one constant (axiom 3: "a size set by its energy alone, more energy
smaller, for volumes and joints alike"), and 0 for an id that is no cell. It reads
`kind` and `cell_quanta` and nothing else. Nothing else in the engine writes `length`.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL
from physics.engine.axiom_3_structure.definitions.size_law import size_law

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _size_kernel(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64), size_constant: wp.float64,
                 length: wp.array(dtype=wp.float32)):
    cell = wp.tid()
    if kind[cell] == NO_CELL:
        length[cell] = wp.float32(0.0)
    else:
        length[cell] = size_law(cell_quanta[cell], size_constant)


def size_cells(state, settings) -> None:
    """Write the length of every cell by the size law."""
    wp.launch(_size_kernel, dim=state.count,
              inputs=[state.kind, state.cell_quanta, wp.float64(settings.size_constant)], outputs=[state.length],
              device=state.device)
