"""Opacity: how solid a dot is, on a fixed log scale of energy density against the start.

Contract: `opacity(density, reference)` returns, per volume, a float64 in
[FAINTEST, 1]. The ratio r = density / reference (reference: the start's mean
density of that volume's kind, `density_reference.py`) is read on a fixed log
scale: r <= THINNEST gives FAINTEST, r >= PACKED gives 1, and in between the log
of the opacity is linear in log r. At r = 1 (the start's mean) a dot is near
invisible (about 0.025), so the start looks like a faint haze and a mass that
grows packed shows up as it packs. The scale never moves with the data: it is not
rank-based, so the same density always gives the same opacity, and a denser
volume is never fainter than a thinner one. A non-finite or non-positive ratio
counts as thinnest.
"""
from __future__ import annotations

import numpy as np

FAINTEST = 0.01
THINNEST = 0.5
PACKED = 16.0


def opacity(density: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per volume, how solid its dot is, from its density over the start's mean."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(density, dtype=float) / np.asarray(reference, dtype=float)
        depth = (np.log(ratio) - np.log(THINNEST)) / (np.log(PACKED) - np.log(THINNEST))
    depth = np.clip(np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return FAINTEST ** (1.0 - depth)
