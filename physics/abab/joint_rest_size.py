"""Joint rest size: the size a joint's energy wants (bible foundation 3: more energy, smaller).

Contract: `joint_rest_size(energy)` returns, elementwise,
  JOINT_SIZE * (JOINT_ENERGY / energy) ** 3
the energy-volume law size ~ energy ** -3 (as in the old cells engine) with the
joint's own scale: a joint holding JOINT_ENERGY (0.05, a twentieth of a background
energy volume's 1) wants JOINT_SIZE (0.03 of a background energy volume's size 1;
rule 8: a joint holds about 2 to 5% of an energy volume). Both constants are picked
from the rule's range, not tuned to a result. A joint pressed smaller than this is
squeezed and pushes back.
"""
from __future__ import annotations

import numpy as np

JOINT_ENERGY = 0.05
JOINT_SIZE = 0.03


def joint_rest_size(energy: np.ndarray | float) -> np.ndarray:
    """Per joint, JOINT_SIZE * (JOINT_ENERGY / energy) ** 3."""
    return JOINT_SIZE * (JOINT_ENERGY / np.asarray(energy, dtype=float)) ** 3
