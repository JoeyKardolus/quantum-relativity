"""Energy flow: one step of the energy fluid's equation on local clocks, every box at its own level.

Contract: `energy_flow(universe, settings)` advances the energy fluid (one complex number
per Planck box, Gross-Pitaevskii) by settings.step_time and returns M, the finest time
level used. The cell sums (cell_sums.py) must be current. It marks the seam boxes
(seam_flags.py), gives every cell its time level (time_levels.py), then runs 2**M
finest substeps, at substep s those boxes and faces due whose level is at least M -
(trailing zeros of s) (all at s = 0). Each substep is a second-order Strang splitting
of exact rotations: the due boxes' on-site turn (on_site.py: pull and push) for half
their dt, the due faces' hops (hop.py: flow; friction at A-B faces; the seal; a joint
only at its seams) along x, y, z in both parities for half, the same hops in reverse
order for the other half, the spreading of the energy friction took in those hops
(spread_heat.py), and the on-site turn again for half. Each box runs for its
own dt = step_time / 2**m times its cell's clock: a packed, slow cell does few long
updates, a fast one many short ones, and no cell waits on the fastest (bible: time is
local). Every rotation keeps |psi|**2 of the boxes it touches, and the spreading
moves energy without making or losing any, so total energy is conserved to float64
rounding. Box owners do not change. universe.substeps grows by
2**M and box_updates by the boxes' own updates. Raises ValueError as time_levels does.
"""
from __future__ import annotations

from physics.potts.hop import hop
from physics.potts.on_site import on_site
from physics.potts.seam_flags import seam_flags
from physics.potts.settings import Settings
from physics.potts.spread_heat import spread_heat
from physics.potts.time_levels import time_levels
from physics.potts.universe import Universe

_FACE_SETS = [(axis, parity) for axis in range(3) for parity in range(2)]


def energy_flow(universe: Universe, settings: Settings) -> int:
    """Advance the energy fluid one step on local clocks; return the finest level."""
    finest = time_levels(universe, settings)
    seam_flags(universe)
    for substep in range(2 ** finest):
        threshold = 0 if substep == 0 else finest - _trailing_zeros(substep)
        on_site(universe, settings, threshold, 0.5)
        for axis, parity in _FACE_SETS:
            hop(universe, settings, threshold, axis, parity, 0.5)
        for axis, parity in reversed(_FACE_SETS):
            hop(universe, settings, threshold, axis, parity, 0.5)
        spread_heat(universe, settings)
        on_site(universe, settings, threshold, 0.5)
    counts = universe.scratch["site_counts"]
    universe.box_updates += int(sum(int(counts[level]) * 2 ** level for level in range(finest + 1)))
    universe.substeps += 2 ** finest
    return finest


def _trailing_zeros(number: int) -> int:
    return (number & -number).bit_length() - 1
