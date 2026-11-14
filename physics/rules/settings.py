"""RuleSettings: which rules run in a step, and how strongly. 0 switches a rule off.

Contract: a frozen dataclass of floats and ints with defaults; every rule in
`step` reads its own field and does nothing when that field is 0. Defaults run
only the energy-length rule, so a default step leaves a consistent network
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSettings:
    """Per-rule strengths for `step` and `rewire`."""

    #: energy-length rule: fraction of the energy*length mismatch removed per step (1 = all).
    energy_length: float = 1.0
    #: energy flow: diffusion rate of energy between connections sharing a joint (0..1).
    energy_flow: float = 0.0
    #: drift: fractional lengthening of every connection per step (energy drops by the same factor).
    drift: float = 0.0
    #: attraction: shortening per step per unit of local energy density (a coupling like G).
    attraction: float = 0.0
    #: joint repulsion: stretching per step per unit of charge at a connection's joints.
    repulsion: float = 0.0
    #: rewiring: local move attempts per step (0 = off).
    rewiring: int = 0
    #: rewiring action: degree every joint is pushed toward.
    target_degree: float = 6.0
    #: rewiring action: triangles per joint every joint is pushed toward.
    target_triangles: float = 0.0
    #: rewiring action: weight of the triangle term against the degree term.
    triangle_weight: float = 1.0
