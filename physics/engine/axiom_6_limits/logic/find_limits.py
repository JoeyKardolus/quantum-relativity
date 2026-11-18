"""Find limits: mark every joint squeezed too far and every energy volume overstretched.

Contract: `find_limits(state, settings)` writes `flags` (COLLAPSE_FLAG on a joint that
`collapse_law` says is past its limit, TEAR_FLAG on an energy volume that `tear_law`
says tears) and `found` (how many of each, [collapses, tears]). It changes nothing
else; `run_events` carries the events out. A volume's three pieces are counted from its
own boxes, across its own swirl axis.
Reads `kind`, `cell_quanta`, `fresh_quanta`, `length`, `swirl_axis`, `quanta`.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, ENERGY_VOLUME, JOINT
from physics.engine.axiom_6_limits.definitions.collapse_law import collapse_law
from physics.engine.axiom_6_limits.definitions.limits_arrays import COLLAPSE_FLAG, TEAR_FLAG
from physics.engine.axiom_6_limits.definitions.tear_law import tear_law

wp.set_module_options({"fuse_fp": False})

_BLOCK = wp.constant(BLOCK)
_PIECE = wp.constant(BLOCK // 3)


@wp.func
def _piece_quanta(quanta: wp.array4d(dtype=wp.int32), cell: wp.int32, axis: wp.int32, piece: wp.int32) -> wp.int64:
    """The quanta in one third of a volume's block: two layers across `axis`."""
    held = wp.int64(0)
    for layer in range(piece * _PIECE, (piece + 1) * _PIECE):
        for first in range(_BLOCK):
            for second in range(_BLOCK):
                if axis == 0:
                    held += wp.int64(quanta[cell, layer, first, second])
                elif axis == 1:
                    held += wp.int64(quanta[cell, first, layer, second])
                else:
                    held += wp.int64(quanta[cell, first, second, layer])
    return held


@wp.kernel
def _mark_limits(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
                 fresh_quanta: wp.array(dtype=wp.int64), length: wp.array(dtype=wp.float32),
                 swirl_axis: wp.array(dtype=wp.int32), quanta: wp.array4d(dtype=wp.int32),
                 flags: wp.array(dtype=wp.int32), found: wp.array(dtype=wp.int32), collapse_thin: wp.float64,
                 tear_length: wp.float64):
    """One thread per cell id: mark a joint past its limit or a volume that is too long."""
    cell = wp.tid()
    if kind[cell] == JOINT:
        if collapse_law(cell_quanta[cell], fresh_quanta[cell], collapse_thin):
            flags[cell] = COLLAPSE_FLAG
            wp.atomic_add(found, 0, 1)
    elif kind[cell] == ENERGY_VOLUME:
        if wp.float64(length[cell]) > tear_length:
            axis = swirl_axis[cell]
            if tear_law(wp.float64(length[cell]), _piece_quanta(quanta, cell, axis, 0),
                        _piece_quanta(quanta, cell, axis, 1), _piece_quanta(quanta, cell, axis, 2), tear_length):
                flags[cell] = TEAR_FLAG
                wp.atomic_add(found, 1, 1)


def find_limits(state, settings) -> None:
    """Mark every joint to collapse and every volume to tear this step."""
    wp.launch(_mark_limits, dim=state.count,
              inputs=[state.kind, state.cell_quanta, state.fresh_quanta, state.length, state.swirl_axis,
                      state.quanta, state.flags, state.found, wp.float64(settings.collapse_thin),
                      wp.float64(settings.tear_length)],
              device=state.device)
