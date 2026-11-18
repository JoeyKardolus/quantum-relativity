"""Balance level: the energy at which each joint's push balances the pull across it.

Contract: `balance_level(state, settings)` (host) returns a float64 array, one number per
cell id in use: for a living joint, the quanta at which its own push (push_law.py:
`P quanta ** 4 / size_constant ** 2`, the length following the quanta by the size law)
equals its press as `press_joints` last wrote it,
  (max(press, 0) * size_constant ** 2 / push_strength) ** (1 / 4)
and 0 for every other id and for a joint nothing presses. It is the level a squeezed
joint goes toward (make_law.py), worked out for every joint at once, so a start laid at
it is at rest in balance: no joint squeezed, none slack. The press depends a little on
the joint's own length, so the start asks again until the answer stands
(`new_universe`). Reads `kind`, `press`; changes nothing.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import JOINT


def balance_level(state, settings) -> np.ndarray:
    """Per cell id: the quanta at which a joint's push equals the press on it; 0 for everything else."""
    press = np.maximum(state.host("press").astype(np.float64), 0.0)
    level = np.power(press * settings.size_constant ** 2 / settings.push_strength, 0.25)
    return np.where(state.host("kind") == JOINT, level, 0.0)
