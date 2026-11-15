"""Hubble gap scene: the growth rate of space early in the run against late in it.

Contract: `hubble_gap(cells_per_side=8, boxes_per_cell_side=6, seed=1, unevenness=0.3,
steps=60, every=2, settings=None, device="cuda")` runs the expansion scene
(expansion.py) with those arguments and returns a dict (no verdict):
  early_rate, late_rate   mean occupied_rate (d ln occupied radius / dt, the size of
                          space) over the first and the last quarter of the rate samples
                          (at least one each)
  late_minus_early        late_rate - early_rate
  late_over_early         late_rate / early_rate (NaN when early_rate is 0)
  known_late_over_early   73.5 / 67.4, the measured late (local) over early (CMB) Hubble
                          rate, km/s per megaparsec, beside which the ratio is read
  expansion               the expansion scene's full result
"""
from __future__ import annotations

import numpy as np

from physics.potts import Settings
from physics.scenes.expansion import expansion

KNOWN_LATE_OVER_EARLY = 73.5 / 67.4


def hubble_gap(cells_per_side: int = 8, boxes_per_cell_side: int = 6, seed: int = 1, unevenness: float = 0.3,
               steps: int = 60, every: int = 2, settings: Settings | None = None, device: str = "cuda") -> dict:
    """Early and late mean growth rate of space in the expansion scene, their gap and ratio."""
    grown = expansion(cells_per_side, boxes_per_cell_side, seed, unevenness, steps, every, settings, device)
    rates = np.asarray(grown["occupied_rate"])
    quarter = max(rates.size // 4, 1)
    early = float(rates[:quarter].mean())
    late = float(rates[-quarter:].mean())
    return {"early_rate": early, "late_rate": late, "late_minus_early": late - early,
            "late_over_early": late / early if early != 0.0 else float("nan"),
            "known_late_over_early": KNOWN_LATE_OVER_EARLY, "expansion": grown}
