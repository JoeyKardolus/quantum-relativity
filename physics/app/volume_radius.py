"""How big a volume is drawn: the radius of a ball holding its size.

Contract: `volume_radius(size)` returns, per volume, the radius of a ball whose volume is
`size` (`(3 size / 4 pi) ** (1/3)`), as float64. The engine's size is a volume; a size that
is not finite or not positive gets radius 0. Display only: the physics never sees it.
"""
from __future__ import annotations

import numpy as np


def volume_radius(size) -> np.ndarray:
    """The radius of a ball of volume `size`, per volume; 0 where the size is not a positive number."""
    sizes = np.asarray(size, dtype=float)
    usable = np.where(np.isfinite(sizes) & (sizes > 0), sizes, 0.0)
    return np.cbrt(3.0 * usable / (4.0 * np.pi))
