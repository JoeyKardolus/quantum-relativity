"""Arrival law: what crosses arrives still moving, less what the joint's friction took.

Contract: `arrival_law(holds, standing, turned, taken, lost)` (Warp func, all float64)
returns the new phase, in (-pi, pi], of a box of the taking mouth.
  holds     the quanta the box held
  standing  its phase
  turned    how far the crossing's step turned the box's fluid (`cross`)
  taken     the quanta arriving in the box
  lost      how many of those lost their direction in the joint (friction_law.py)
What stood there and what arrived with its direction go on as the step left them: at
`standing + turned`. The quanta whose direction the joint took arrive at rest in the
fluid they join: at `standing`. The two join as fluids do everywhere (joined_phase.py,
axiom 7): each counts by its quanta, and the box gets the angle of the two added (the
standing phase where they cancel). So friction takes motion and never makes any: the
more of what arrives is lost, the less the box turns, and nothing is ever turned further
than the step turned it. No angle is drawn anywhere.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_7_quantum.definitions.joined_phase import CANCELLED
from physics.engine.axiom_7_quantum.definitions.wrap_phase import wrap_phase

wp.set_module_options({"fuse_fp": False})

_CANCELLED = wp.constant(wp.float64(CANCELLED))


@wp.func
def arrival_law(holds: wp.float64, standing: wp.float64, turned: wp.float64, taken: wp.float64,
                lost: wp.float64) -> wp.float64:
    """The phase of a taking mouth box: its moving fluid at the turned phase joined with the quanta at rest."""
    moving = holds + taken - lost
    real = moving * wp.cos(standing + turned) + lost * wp.cos(standing)
    imaginary = moving * wp.sin(standing + turned) + lost * wp.sin(standing)
    if real * real + imaginary * imaginary <= _CANCELLED * _CANCELLED * (moving + lost) * (moving + lost):
        return wrap_phase(standing)
    return wrap_phase(wp.atan2(imaginary, real))
