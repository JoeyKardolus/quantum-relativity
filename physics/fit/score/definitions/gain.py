"""Gain: how much a change since the start counts, capped, and nothing for a loss.

Contract: `gain(change, scale)` returns `tanh(change / scale)` for a change above 0 and
0.0 otherwise: between 0 and 1, so no runaway earns more than a steady gain does.
"""
from __future__ import annotations

import math


def gain(change: float, scale: float) -> float:
    """tanh(change / scale), 0 for a loss."""
    return math.tanh(max(change, 0.0) / scale)
