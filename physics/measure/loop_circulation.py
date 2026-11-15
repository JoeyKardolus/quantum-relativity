"""Loop circulation: the phase winding and the flow's circulation round a square loop of boxes.

Contract: `loop_circulation(field, centre, normal, half_width)` takes a complex numpy
array (nx, ny, nz), a point (3 floats, box units), a grid axis `normal` (0, 1 or 2) and
a whole half width >= 1. The loop is the square of boxes in the layer through the
centre's nearest box along `normal`, corners at the centre's nearest box plus and minus
half_width along the two other axes, walked right-handed about +normal (as
phase_winding.py). Returns a dict:
  winding          the phase change round the loop (each box-to-box change wrapped
                   into (-pi, pi]) over 2 pi: a whole number by construction (int)
  flow_circulation the flow's circulation round the loop over 2 pi, the flow on a step
                   from box i to j being the lattice current Im(conj(psi_i) psi_j) over
                   the mean density (|psi_i|^2 + |psi_j|^2) / 2, which is sin of the phase
                   step where the density is even; a whole-step whirl gives 1 only when
                   the phase changes slowly round the loop, so how far it sits from a
                   whole number says how well the grid resolves the whirl
  boxes            the loop's length in box steps (8 half_width)
Raises ValueError when the loop leaves the grid, half_width < 1, or a loop box is 0.
"""
from __future__ import annotations

import numpy as np


def loop_circulation(field: np.ndarray, centre, normal: int, half_width: int) -> dict:
    """Whole-turn phase winding and flow circulation round a square loop about a point."""
    field = np.asarray(field, dtype=np.complex128)
    if half_width < 1 or normal not in (0, 1, 2):
        raise ValueError(f"need half_width >= 1 and normal in 0, 1, 2, got {half_width} and {normal}")
    middle = np.rint(np.asarray(centre, dtype=np.float64)).astype(np.int64)
    path = _square(middle, normal, half_width)
    if path.min() < 0 or np.any(path.max(axis=0) >= np.array(field.shape)):
        raise ValueError(f"the loop round {middle.tolist()} of half width {half_width} leaves the grid")
    values = field[tuple(path.T)]
    if np.any(values == 0):
        raise ValueError("a box on the loop holds 0, so it has no phase")
    following = np.roll(values, -1)
    product = np.conj(values) * following
    density = 0.5 * (np.abs(values) ** 2 + np.abs(following) ** 2)
    return {"winding": int(np.rint(np.angle(product).sum() / (2.0 * np.pi))),
            "flow_circulation": float((product.imag / density).sum() / (2.0 * np.pi)),
            "boxes": int(len(path))}


def _square(middle: np.ndarray, normal: int, half_width: int) -> np.ndarray:
    """The loop's boxes in walking order: +across, +up, -across, -up, starting at the low corner."""
    across, up = (normal + 1) % 3, (normal + 2) % 3
    side = 2 * half_width
    steps = [(across, 1)] * side + [(up, 1)] * side + [(across, -1)] * side + [(up, -1)] * side
    place = middle.copy()
    place[across] -= half_width
    place[up] -= half_width
    path = []
    for axis, sign in steps:
        path.append(place.copy())
        place[axis] += sign
    return np.array(path)
