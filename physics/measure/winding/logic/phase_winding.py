"""Phase winding: how many whole turns the phase makes round every smallest loop of boxes.

Contract: `phase_winding(field, valid=None)` takes a complex numpy array of shape
(nx, ny, nz), one complex number per Planck box (amount times exp(i phase)), and an
optional boolean mask of the same shape (boxes outside it, and boxes whose number is
exactly 0, have no phase). It returns an int64 array of shape (3, nx, ny, nz): entry
[axis, x, y, z] is the winding round the unit square of boxes whose lowest corner is
(x, y, z) and whose normal is `axis`, walked right-handed (definitions/in_plane.py).
Each of the four steps is the short way round (definitions/phase_step.py), so their
sum is a whole number of full turns: +n where a swirl of n turns pierces the square
along +axis, 0 where none does. Squares running off the array or touching a box without
phase hold 0. The array is one cell's block, or a made-up test field: boxes are not
space, so no loop here ever spans two cells.
"""
from __future__ import annotations

import numpy as np

from physics.measure.winding.definitions.in_plane import AXES, in_plane
from physics.measure.winding.definitions.phase_step import phase_step

_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))


def _corner(values: np.ndarray, across: int, step_across: int, up: int, step_up: int) -> np.ndarray:
    """The values one corner of every square sees: shifted 0 or 1 along the two in-plane axes."""
    pick = [slice(None)] * AXES
    pick[across] = slice(step_across, values.shape[across] - 1 + step_across)
    pick[up] = slice(step_up, values.shape[up] - 1 + step_up)
    return values[tuple(pick)]


def _square_turns(field: np.ndarray, has_phase: np.ndarray, normal: int) -> np.ndarray:
    """The whole turns round every unit square with one normal; 0 where a corner has no phase."""
    across, up = in_plane(normal)
    corners = [_corner(field, across, step_across, up, step_up) for step_across, step_up in _CORNERS]
    present = [_corner(has_phase, across, step_across, up, step_up) for step_across, step_up in _CORNERS]
    turned = sum(phase_step(corners[index], corners[(index + 1) % 4]) for index in range(4))
    whole = np.rint(turned / (2.0 * np.pi)).astype(np.int64)
    whole[~(present[0] & present[1] & present[2] & present[3])] = 0
    return whole


def phase_winding(field: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Whole turns of the phase round every unit square of boxes, per normal axis."""
    field = np.asarray(field, dtype=np.complex128)
    if field.ndim != AXES:
        raise ValueError(f"field must be 3D, got shape {field.shape}")
    has_phase = field != 0
    if valid is not None:
        has_phase &= np.asarray(valid, dtype=bool)
    windings = np.zeros((AXES,) + field.shape, dtype=np.int64)
    for normal in range(AXES):
        across, up = in_plane(normal)
        target = [slice(None)] * AXES
        target[across] = slice(0, field.shape[across] - 1)
        target[up] = slice(0, field.shape[up] - 1)
        windings[normal][tuple(target)] = _square_turns(field, has_phase, normal)
    return windings
