"""Rehome site (Warp func): the Planck box a rehome thread works on, in its own universe's coordinates.

Contract: `rehome_site(home, ti, tj, tk, phase_index, side)` returns (site, valid): the
box home * side + ox + 3 ti, oy + 3 tj, oz + 3 tk of the stacked arrays (state.py), with
(ox, oy, oz) = (phase_index // 9, phase_index // 3 % 3, phase_index % 3), and whether it
lies inside universe `home`. Sites of one phase lie 3 apart in each universe's own
coordinates, so the 3 x 3 x 3 blocks round two sites never overlap (drain.py,
hand_over.py), and a universe's sites are the same alone or side by side in a batch.
`rehome_dims(state, phase_index)` is the launch shape on the host.
"""
from __future__ import annotations

import warp as wp


@wp.func
def rehome_site(home: int, ti: int, tj: int, tk: int, phase_index: int, side: int):
    """The box a rehome thread works on and whether it lies inside its universe."""
    local = wp.vec3i(phase_index // 9 + 3 * ti, phase_index // 3 % 3 + 3 * tj, phase_index % 3 + 3 * tk)
    valid = local[0] < side and local[1] < side and local[2] < side
    return wp.vec3i(home * side + local[0], local[1], local[2]), valid


def rehome_dims(state, phase_index: int) -> tuple:
    """The launch shape of rehome phase `phase_index`: (universes, sites along each axis)."""
    offset = (phase_index // 9, phase_index // 3 % 3, phase_index % 3)
    return (state.universes,) + tuple(max(0, -(-(state.side - start) // 3)) for start in offset)
