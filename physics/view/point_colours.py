"""Point colours: one RGBA byte quad per volume, red for energy volumes, blue for joints.

Contract: `point_colours(kind, density, reference)` returns an (n, 4) uint8 array.
RGB is RED for KIND_A and BLUE for KIND_B; alpha is `opacity(density,
reference[kind])` (fixed log scale against the start's mean of that kind,
`opacity.py`) times 255, rounded. The drawing blends additively, so overlapping
faint dots add up the way packed energy does.
"""
from __future__ import annotations

import numpy as np

from physics.abab.state import KIND_B
from physics.view.opacity import opacity

RED = (255, 64, 38)
BLUE = (64, 115, 255)


def point_colours(kind: np.ndarray, density: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """RGBA bytes per volume: kind colour, opacity from density over the start's mean."""
    kind = np.asarray(kind)
    is_joint = kind == KIND_B
    colours = np.empty((kind.size, 4), dtype=np.uint8)
    colours[:, :3] = np.where(is_joint[:, None], BLUE, RED)
    alpha = opacity(density, np.asarray(reference, dtype=float)[is_joint.astype(np.int64)])
    colours[:, 3] = np.rint(alpha * 255.0).astype(np.uint8)
    return colours
