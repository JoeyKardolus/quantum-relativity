"""One recorded frame: what the page draws for one step, plus what it measured.

Contract: `Frame` is a frozen value. `positions` (joint_count, 3) is a display
layout only, never part of the network. `energy` is each joint's energy
(joint_energy.py); `bright` the (k, 2) joint pairs of the highest-energy
connections, drawn as lines; `marked` the scene's followed joints renumbered
into this frame; `marked_gap` the network distance between the first two
marked joints (None with fewer than two); `mean_length` the mean connection
length; `background_length` the mean length of the connections inside the far
background (`background_joints`), the ruler for the size of space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Display positions, joint energies and measurements for one step."""

    step: int
    joint_count: int
    positions: np.ndarray
    energy: np.ndarray
    bright: np.ndarray
    marked: tuple[int, ...]
    marked_gap: float | None
    mean_length: float
    background_length: float
