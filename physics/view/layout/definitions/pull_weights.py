"""Pull weights: how far a cell's size pulls each corner plane along its row.

Contract: `pull_weights(reach)` returns float64 (2 * reach,): weight[t] is the share of
a slot's width change that moves a plane lying `reach - t - 1/2` places after it
(negative: before it). A plane right after a slot moves by half the change, one right
before it by minus half, so the slot itself takes the whole change; the pull then
falls smoothly (a raised cosine) to nothing at `reach` places, and a plane is pulled
the opposite way by slots on its other side. The weights of mirrored places cancel.
"""
from __future__ import annotations

import numpy as np


def pull_weights(reach: int) -> np.ndarray:
    """The share of a slot's width change each of the 2 * reach nearest planes moves by."""
    offset = reach - np.arange(2 * reach) - 0.5
    fade = 0.5 * (1.0 + np.cos(np.pi * (np.abs(offset) - 0.5) / reach))
    return 0.5 * np.sign(offset) * fade
