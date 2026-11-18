"""Swirl pattern: one whole swirl in a cell's block, and a box's whole share of it.

Across the axis the wave is the standing mode along one edge plus the same mode a
quarter turn behind (or ahead: the other sense) along the other. The phase so goes
once round the axis, which forces the wave to 0 on it: the swirl holds its own core
open by its turning. Three axes, two senses.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_2_swirls.definitions.standing_mode import standing_mode


def swirl_patterns(across: int) -> np.ndarray:
    """The six patterns as (6, boxes, 2): each box's share (mean 1) and its phase."""
    mode = standing_mode(across)[0]
    patterns = np.empty((6, across ** 3, 2))
    for laid in range(6):
        axis = laid // 2
        sense = 1 if laid % 2 == 0 else -1
        layer = mode[:, None] - 1j * sense * mode[None, :]    # the swirl across its axis
        block = np.repeat(layer[:, :, None], across, axis=2)  # the same layer all along it
        # the block's third direction becomes the swirl's axis, the layer's two the others
        wave = np.moveaxis(block, (0, 1, 2), ((axis + 1) % 3, (axis + 2) % 3, axis)).ravel()
        patterns[laid, :, 0] = np.abs(wave) ** 2 / np.mean(np.abs(wave) ** 2)
        patterns[laid, :, 1] = np.angle(wave)
    return patterns


@wp.func
def dealt(total: wp.int32, share: wp.float64, boxes: wp.int32) -> wp.int32:
    """A box's whole share of its cell's total, rounded down."""
    return wp.int32(wp.floor(wp.float64(total) * share / wp.float64(boxes)))
