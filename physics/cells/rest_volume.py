"""The energy-volume rule (bible rule 4): size follows energy density, length = VOLUME_SCALE / energy per quantum.

Contract: `rest_volume(energy, quanta)` gives the volume an energy volume takes when
nothing squeezes or stretches it, elementwise:
  quanta * (VOLUME_SCALE * quanta / energy) ** 3
An energy volume holds `quanta` quanta of space; each quantum holds energy
eps = energy / quanta and is a cube of side VOLUME_SCALE / eps (length = constant /
energy, like E = hc / lambda). So the rest density energy / volume is
eps ** 4 / VOLUME_SCALE ** 3: size follows the energy per quantum, a density, not
the energy one label holds. A split shares energy and quanta in the same shares
(by volume), so every child keeps eps and the children's rest volumes add up to the
parent's: splitting only redraws the labels. The leak moves energy, not quanta, so a
volume that gains energy gets smaller and more packed (volume ~ energy ** -3 while
its quanta stay), one that loses grows. A joint collapse hands energy and quanta
on together, so the receiving volume's eps becomes the quanta-weighted mean and
its rest volume is at most the two rest volumes summed (more packed).
VOLUME_SCALE (1.0) is the rule's one constant; the background eps is 1, so a
background quantum has volume 1.
"""
from __future__ import annotations

import numpy as np

VOLUME_SCALE = 1.0


def rest_volume(energy: np.ndarray | float, quanta: np.ndarray | float) -> np.ndarray:
    """Volume of an energy volume holding this energy in these quanta: q * (VOLUME_SCALE * q / E) ** 3."""
    held = np.asarray(quanta, dtype=float)
    return held * (VOLUME_SCALE * held / np.asarray(energy, dtype=float)) ** 3
