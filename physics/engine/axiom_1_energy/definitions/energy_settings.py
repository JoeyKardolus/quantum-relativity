"""Energy settings: the numbers axiom 1 leaves open, the step and the start's energies.

Contract: `EnergySettings` is a frozen record; every default was chosen before any run.
  step_time         0.25: counter time per step; a cell's own time per step is this
                    times its clock rate
  volume_quanta     3000: the start's mean quanta per box of an energy volume
  joint_quanta      100: the start's mean quanta per box of a joint
  unevenness        0.1: how much the start's energy differs from volume to volume
  slope             -3.0: how that difference is spread over long and short distances
                    (`uneven_field`); any number
  inner_unevenness  0.1: how much it differs from box to box inside one cell
  joint_unevenness  0.0: how much it differs from joint to joint
`check_energy()` raises ValueError when `step_time`, `volume_quanta`
or `joint_quanta` is not above 0, or an unevenness is below 0 (an even start is a start,
not a rule switched off); the record runs it when it is made, and the engine's
`Settings` runs it again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EnergySettings:
    """The step's length and the start's energies."""

    step_time: float = 0.25
    volume_quanta: float = 3000.0
    joint_quanta: float = 100.0
    unevenness: float = 0.1
    slope: float = -3.0
    inner_unevenness: float = 0.1
    joint_unevenness: float = 0.0

    def __post_init__(self) -> None:
        self.check_energy()

    def check_energy(self) -> None:
        """Refuse a step or start energy that is not above 0 and an unevenness below 0."""
        for name in ("step_time", "volume_quanta", "joint_quanta"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be above 0, got {getattr(self, name)}")
        for name in ("unevenness", "inner_unevenness", "joint_unevenness"):
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"{name} must be 0 or more, got {getattr(self, name)}")
        if not math.isfinite(self.slope):
            raise ValueError(f"slope must be a number, got {self.slope}")
