"""Cell colours: red energy volumes and blue joints, more solid the more packed.

Contract: `cell_colours(kind, relative)` returns uint8 (n, 4) RGBA. The hue is the
kind: an energy volume VOLUME_RED, anything else JOINT_BLUE. The opacity (alpha / 255)
is `opacity_law` of the packing over the start's mean, so a lump turns solid as it
gathers energy and a thinned cell fades to nothing. This is the opacity of one layer of
cells over black (the slice); the whole grid raises it to WHOLE_POWER when it draws.
"""
from __future__ import annotations

import numpy as np

from physics.view.colours.definitions.opacity_law import opacity_law
from physics.view.colours.definitions.palette import JOINT_BLUE, VOLUME_RED
from physics.view.layout.definitions.kinds import ENERGY_VOLUME


def cell_colours(kind: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """RGBA of every cell: the hue from its kind, the opacity from its packing."""
    opacity = opacity_law(relative)
    colours = np.empty((len(opacity), 4), dtype=np.uint8)
    colours[:, :3] = np.where((np.asarray(kind) == ENERGY_VOLUME)[:, None], VOLUME_RED, JOINT_BLUE)
    colours[:, 3] = np.round(255.0 * opacity)
    return colours
