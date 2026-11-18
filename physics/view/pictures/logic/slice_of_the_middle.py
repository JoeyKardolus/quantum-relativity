"""Slice of the middle: the picture `saved_frames` takes when no GPU draws.

Contract: `slice_of_the_middle(pieces, low, high)` returns `slice_picture` of the layer
`middle_layer` names, where energy volumes lie. No GPU, no window.
"""
from __future__ import annotations

import numpy as np

from physics.view.drawing.definitions.middle_layer import middle_layer
from physics.view.layout.definitions.pieces import Pieces
from physics.view.pictures.logic.slice_picture import slice_picture


def slice_of_the_middle(pieces: Pieces, low, high) -> np.ndarray:
    """The slice through the middle layer as an image."""
    return slice_picture(pieces, middle_layer(pieces.side), low, high)
