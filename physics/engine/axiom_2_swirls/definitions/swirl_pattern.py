"""Swirl pattern: the ball whirl in a cell's block, and a box's whole share of it.

The wave is the standing mode along one direction times the same mode along the next, for
each of the three pairs of directions, the three pairs a third of a turn apart in phase.
The three are standing waves of the step with one turn, so a step only turns their sum,
by twice the mode's turn. The wave is 0 on the line through the middle along each of the
three directions, and the phase goes once round each of those lines the same way: one
whirl, the same seen along all three. Nothing else of this kind treats the three alike.
(It is 0 on one long diagonal too, where the phase goes round the other way.)
A cell is laid as it is (0) or turned half round about direction 0, 1 or 2 (1, 2, 3).
"""
import numpy as np
import warp as wp

from physics.engine.axiom_2_swirls.definitions.standing_mode import standing_mode

THIRD = np.exp(2j * np.pi / 3)  # a third of a turn of phase


def swirl_patterns(across: int) -> np.ndarray:
    """The four layings as (4, boxes, 2): each box's share (mean 1) and its phase."""
    mode = standing_mode(across)[0]
    patterns = np.empty((4, across ** 3, 2))
    for laid in range(4):
        # half round about a direction is the mode the other way along it: the whirl is
        # the same from its two ends
        rows = [mode * (-1 if laid == way + 1 else 1) for way in range(3)]
        first, second, third = rows[0][:, None, None], rows[1][None, :, None], rows[2][None, None, :]
        wave = (first * second + THIRD * second * third + THIRD ** 2 * third * first).ravel()
        patterns[laid, :, 0] = np.abs(wave) ** 2 / np.mean(np.abs(wave) ** 2)
        patterns[laid, :, 1] = np.angle(wave)
    return patterns


@wp.func
def dealt(total: wp.int32, share: wp.float64, boxes: wp.int32) -> wp.int32:
    """A box's whole share of its cell's total, rounded down."""
    return wp.int32(wp.floor(wp.float64(total) * share / wp.float64(boxes)))
