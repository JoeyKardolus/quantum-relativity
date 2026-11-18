"""Wound block: a block of boxes whose phase goes a known number of turns round one of its axes.

Contract: `wound_block(axis, turns, side=6, middle=(2.4, 2.6))` returns `(quanta, phase)`,
two (side, side, side) arrays: 100 quanta in every box, and a phase of `turns` times the
angle of the box round the line along `axis` through `middle` (its two coordinates
across the axis, in right-handed order), brought into (-pi, pi]. The same in every
layer. Right-handed about +axis it reads `turns` whole turns. The middle is off the
block's own middle so that no step is exactly half a turn.
"""
from __future__ import annotations

import numpy as np

from physics.measure.winding.definitions.in_plane import in_plane


def wound_block(axis: int, turns: int, side: int = 6, middle: tuple = (2.4, 2.6)) -> tuple:
    """Even quanta and a phase wound `turns` times, right-handed, round `axis`."""
    places = np.stack(np.meshgrid(*[np.arange(side, dtype=np.float64)] * 3, indexing="ij"), axis=-1)
    across, up = in_plane(axis)
    angle = np.arctan2(places[..., up] - middle[1], places[..., across] - middle[0])
    phase = np.angle(np.exp(1j * turns * angle))
    return np.full((side,) * 3, 100, dtype=np.int32), phase
