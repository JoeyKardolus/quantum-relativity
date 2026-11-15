"""Phase winding: how many whole turns the phase of a complex field makes round every smallest loop of boxes.

Contract: `phase_winding(field, valid=None)` takes a complex numpy array of shape
(nx, ny, nz), one complex number per Planck box, and an optional boolean mask of the
same shape (boxes outside it, and boxes whose number is exactly 0, have no phase).
It returns an int64 array of shape (3, nx, ny, nz): entry [axis, x, y, z] is the
winding round the unit square of boxes whose lowest corner is (x, y, z) and whose
normal is `axis`, walked +b, +c, -b, -c with b = (axis + 1) % 3, c = (axis + 2) % 3
(right-handed about the normal). Each step adds the phase change wrapped into
(-pi, pi]; the sum over the four steps is a whole number of 2 pi, so a winding is
exact by construction: +n where a whole-step whirl of n turns pierces the square
along +axis, 0 where none does. Squares running off the grid or touching a box
without phase hold 0.
"""
from __future__ import annotations

import numpy as np


def phase_winding(field: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Whole turns of the phase round every unit square of boxes, per normal axis."""
    field = np.asarray(field, dtype=np.complex128)
    if field.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {field.shape}")
    has_phase = field != 0
    if valid is not None:
        has_phase &= np.asarray(valid, dtype=bool)
    windings = np.zeros((3,) + field.shape, dtype=np.int64)
    for axis in range(3):
        across, up = (axis + 1) % 3, (axis + 2) % 3
        corner = [_shifted(field, across, 0, up, 0), _shifted(field, across, 1, up, 0),
                  _shifted(field, across, 1, up, 1), _shifted(field, across, 0, up, 1)]
        present = [_shifted(has_phase, across, 0, up, 0), _shifted(has_phase, across, 1, up, 0),
                   _shifted(has_phase, across, 1, up, 1), _shifted(has_phase, across, 0, up, 1)]
        turn = sum(np.angle(np.conj(corner[index]) * corner[(index + 1) % 4]) for index in range(4))
        whole = np.rint(turn / (2.0 * np.pi)).astype(np.int64)
        whole[~(present[0] & present[1] & present[2] & present[3])] = 0
        target = [slice(None)] * 3
        target[across] = slice(0, field.shape[across] - 1)
        target[up] = slice(0, field.shape[up] - 1)
        windings[axis][tuple(target)] = whole
    return windings


def _shifted(values: np.ndarray, across: int, step_across: int, up: int, step_up: int) -> np.ndarray:
    """The values one corner of every square sees: shifted 0 or 1 along the two in-plane axes."""
    pick = [slice(None)] * 3
    pick[across] = slice(step_across, values.shape[across] - 1 + step_across)
    pick[up] = slice(step_up, values.shape[up] - 1 + step_up)
    return values[tuple(pick)]
