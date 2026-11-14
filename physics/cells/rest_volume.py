"""The energy-volume rule (bible rule 4): length = VOLUME_SCALE / energy, volume = length^3.

Contract: `rest_volume(energy)` gives the volume an energy volume takes when
nothing squeezes or stretches it: (VOLUME_SCALE / energy) ** 3, elementwise. More
energy, smaller and more packed: energy per volume grows as energy ** 4.
VOLUME_SCALE (1.0) is the rule's one constant; the background energy is 1, so an
even grid starts with every volume 1.
"""
from __future__ import annotations

import numpy as np

VOLUME_SCALE = 1.0


def rest_volume(energy: np.ndarray | float) -> np.ndarray:
    """Volume of an energy volume holding this energy: (VOLUME_SCALE / energy) ** 3."""
    return (VOLUME_SCALE / np.asarray(energy, dtype=float)) ** 3
