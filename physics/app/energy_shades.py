"""Per-connection energy as a colour value between 0 and 1.

Contract: `energy_shades(network)` returns one value per connection:
log10 of its energy, mapped linearly so the frame's lowest energy is 0 and its
highest is 1, rounded to 3 decimals. The map is redone every frame, so the
colours show where energy sits relative to the rest, not how the whole
network's energy changes (mean_length_vs_start shows that). A spread smaller
than MIN_DECADES is drawn over MIN_DECADES, so float noise on even space is
not blown up into full colour.
"""
from __future__ import annotations

import numpy as np

from physics.network import Network

MIN_DECADES = 0.01


def energy_shades(network: Network) -> list[float]:
    """log10 energy per connection, scaled to 0..1 over this frame's range."""
    if not network.connection_count:
        return []
    decades = np.log10(network.energy)
    lowest = float(decades.min())
    span = max(float(decades.max()) - lowest, MIN_DECADES)
    return np.round(np.clip((decades - lowest) / span, 0.0, 1.0), 3).tolist()
