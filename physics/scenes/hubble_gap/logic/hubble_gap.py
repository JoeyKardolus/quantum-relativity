"""Hubble gap scene: the growth rate of the universe early in the run against late in it.

Contract: `hubble_gap(cells_per_side=16, seed=1, steps=200, every=10, settings=None,
device="cuda", start=None)` runs the expansion scene with those arguments and returns a
dict (no verdict):
  early_rate, late_rate   mean size_rate (d ln size / dt) over the first and the last
                          quarter of the rate samples (early_late)
  late_minus_early        late_rate - early_rate
  late_over_early         late_rate / early_rate (NaN when early_rate is 0)
  known_late_over_early   73.5 / 67.4, beside which the ratio is read
  expansion               the expansion scene's full result
"""
from __future__ import annotations

from physics.engine import Settings
from physics.scenes.expansion.logic.expansion import expansion
from physics.scenes.hubble_gap.definitions.early_late import early_late
from physics.scenes.hubble_gap.definitions.known import KNOWN_LATE_OVER_EARLY


def hubble_gap(cells_per_side: int = 16, seed: int = 1, steps: int = 200, every: int = 10,
               settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """Early and late mean growth rate of the universe in the expansion scene, their gap and ratio."""
    grown = expansion(cells_per_side, seed, steps, every, settings, device, start)
    early, late = early_late(grown["size_rate"])
    return {"early_rate": early, "late_rate": late, "late_minus_early": late - early,
            "late_over_early": late / early if early != 0.0 else float("nan"),
            "known_late_over_early": KNOWN_LATE_OVER_EARLY, "expansion": grown}
