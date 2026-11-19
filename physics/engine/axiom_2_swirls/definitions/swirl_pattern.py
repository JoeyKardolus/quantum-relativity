"""Swirl pattern: a cell's three ring whirls, and a box's whole share of the cell.

Every ring of boxes round a core holds the ring's standing whirl (`standing_mode.py`): one
whole turn, the same in all three rings, at every depth and all along each ring. So the
three directions are alike box for box, every box holds fluid and all of it moves; the
empty core is where no box is. A step only turns it. The sign is the direction of the
turn: a joint's whirl (1) is a volume's (0) seen in a mirror, the boxes round each core in
the opposite order. No cell is laid by where it stands.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_2_swirls.definitions.standing_mode import standing_mode


def swirl_patterns(across: int) -> np.ndarray:
    """A volume's and a joint's whirl as (2, boxes, 2): each box's share (mean 1) and its phase."""
    mode = standing_mode(across)[0]
    patterns = np.empty((2, across ** 3, 2))
    for minus in range(2):
        around = mode[::-1] if minus else mode  # the mirror turns the other way
        wave = np.broadcast_to(around[None, :, None], (across,) * 3).ravel()
        patterns[minus, :, 0] = np.abs(wave) ** 2 / np.mean(np.abs(wave) ** 2)
        patterns[minus, :, 1] = np.angle(wave)
    return patterns


@wp.func
def dealt(total: wp.int32, share: wp.float64, boxes: wp.int32) -> wp.int32:
    """A box's whole share of its cell's total, rounded down."""
    return wp.int32(wp.floor(wp.float64(total) * share / wp.float64(boxes)))
