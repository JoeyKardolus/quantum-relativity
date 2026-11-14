"""CellSettings: the strength of each cell rule; every rule is always on in a scene.

Contract: frozen record of floats, read by step():
  attraction   squeeze a joint gets per unit of pull potential (rule 7)
  joint_push   push a joint gives back per unit of squeeze (rule 8)
  push_limit   the most push a joint can give; asked for more, it collapses
  drift        fractional growth of every volume per step (rule 6)
  energy_flow  rate energy moves between neighbours through their joints (rule 5)
`CellSettings.all_on(**overrides)` gives the defaults below, drift 0 (fill it
with balancing_drift):
  attraction   0.005  even background (potential 10) squeezes each joint by 0.05
  joint_push   0.5    a joint pushes back half the squeeze it gets
  push_limit   0.05   twice the push at background squeeze: a joint squeezed
                      twice as hard as the background gives way
  energy_flow  0.05   slow evening-out, so lumps can grow against it
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CellSettings:
    """Strengths of attraction, joint push, push limit, drift and energy flow."""

    attraction: float
    joint_push: float
    push_limit: float
    drift: float
    energy_flow: float

    @classmethod
    def all_on(cls, **overrides: float) -> "CellSettings":
        """Defaults with every rule on (drift 0 until balanced), plus the given overrides."""
        return replace(_ALL_ON, **overrides)


_ALL_ON = CellSettings(attraction=0.005, joint_push=0.5, push_limit=0.05, drift=0.0,
                       energy_flow=0.05)
