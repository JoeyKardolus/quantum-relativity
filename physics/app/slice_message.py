"""The "slice" message: a plane cut through the display layout, drawn as foam. Display only.

Contract: `slice_message(positions, radius, energy, joints, squeeze, depth,
axis, step)` cuts the (n, 3) display positions with the plane square to
`axis` ("x", "y" or "z") at `depth` (0..1 across the positions' extent on
that axis). An energy volume crosses the plane when its slice weight
radius^2 - distance^2 is positive; the crossing volumes' 2D power diagram
(`power_cells`, weights = slice weights) inside the view box (the positions'
extent on the two plane axes: x cut -> (y, z), y cut -> (x, z), z cut ->
(x, y)) is the slice of the 3D power diagram, so the cells tile the box. Returns
{"type": "slice", "step", "depth", "cells": [{"id", "polygon": [x0, y0, ...],
"energy"}], "walls": [{"a", "b", "line": [x0, y0, x1, y1], "squeeze",
"band"}]}; "energy" is the volume's entry in `energy`, a wall's "squeeze" is
the entry in `squeeze` of the joint between a and b, or null when a and b are
not neighbours in the grid (a wall the drawing needs that no joint matches).
"band" is the width of the joint layer on that wall: the joint's entry in
`bands` (the thickness of its 3D joint layer, `walls_message`), null for a
wall that is no joint, for a joint without a 3D face, or when `bands` is None.
The page draws the band centred on the line, in its own colour.
Coordinates have 3 decimals. Nothing here is stored on the Grid.
"""
from __future__ import annotations

import numpy as np

from physics.app.joint_index import joint_index
from physics.app.power_cells import power_cells

PLANE_AXES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
CUT_AXIS = {"x": 0, "y": 1, "z": 2}


def slice_message(positions: np.ndarray, radius: np.ndarray, energy, joints: np.ndarray,
                  squeeze, depth: float, axis: str, step: int, bands=None) -> dict:
    """The foam cut through the display layout at depth along axis."""
    if axis not in CUT_AXIS:
        raise ValueError(f"axis must be x, y or z, not {axis!r}")
    message = {"type": "slice", "step": int(step), "depth": round(float(depth), 3),
               "cells": [], "walls": []}
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if not len(positions):
        return message
    across = positions[:, CUT_AXIS[axis]]
    plane = across.min() + float(depth) * (across.max() - across.min())
    weight = np.asarray(radius, dtype=float) ** 2 - (across - plane) ** 2
    crossing = np.flatnonzero(weight > 0)
    if not len(crossing):
        return message
    flat = positions[:, PLANE_AXES[axis]]
    box = (*flat.min(axis=0), *flat.max(axis=0))
    polygons, walls = power_cells(flat[crossing], weight[crossing], box)
    message["cells"] = [{"id": int(crossing[index]), "polygon": _rounded(polygon),
                         "energy": energy[crossing[index]]}
                        for index, polygon in enumerate(polygons) if polygon is not None]
    joint_of = joint_index(joints, len(positions),
                           [(crossing[first], crossing[second]) for first, second, _ in walls])
    message["walls"] = [{"a": int(crossing[first]), "b": int(crossing[second]),
                         "line": _rounded(segment),
                         "squeeze": None if joint < 0 else squeeze[joint],
                         "band": None if joint < 0 or bands is None else bands[joint]}
                        for (first, second, segment), joint in zip(walls, joint_of)]
    return message


def _rounded(points):
    return np.round(points, 3).ravel().tolist()

