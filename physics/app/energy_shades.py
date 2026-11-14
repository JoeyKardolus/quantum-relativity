"""Per energy volume, its energy per unit volume as a colour value between 0 and 1.

Contract: `energy_shades(grid)` returns one value per energy volume: log10 of
energy / volume, mapped linearly so the frame's lowest is 0 and its highest is
1, rounded to 3 decimals. The map is redone every frame, so the colours show
where energy is packed relative to the rest, not how the whole grid changes
(size_vs_start shows that). A spread smaller than MIN_DECADES is drawn over
MIN_DECADES, so float noise on an even grid is not blown up into full colour.
"""
from __future__ import annotations

import numpy as np

from physics.cells import Grid

MIN_DECADES = 0.01


def energy_shades(grid: Grid) -> list[float]:
    """log10 energy per volume for each energy volume, scaled to 0..1 over this frame."""
    if not grid.cell_count:
        return []
    decades = np.log10(grid.energy / grid.volume)
    lowest = float(decades.min())
    span = max(float(decades.max()) - lowest, MIN_DECADES)
    return np.round(np.clip((decades - lowest) / span, 0.0, 1.0), 3).tolist()
