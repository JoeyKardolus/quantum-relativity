"""Whirl strength: the swirl of a complex field round a line, as angular momentum per unit density.

Contract: `whirl_strength(field, centre, axis, radius, half_height, valid=None)` takes a complex
numpy array (nx, ny, nz) (one number per Planck box), a point and a direction (3
floats each, box units; the direction need not be unit length) and two distances. Over
the boxes within `radius` of the line through `centre` along `axis` and within
`half_height` of the centre along it, it returns the angular momentum about the line
per unit density, sum Im(conj(psi) (r x grad psi)) . axis / sum |psi|^2, with r the
box's place from the centre and grad by central differences (one-sided at the grid's
faces): in units of whole steps (hbar = 1, mass = 1), so a whirl of n whole turns centred
on the line gives about n, less as it drifts off the line or the grid fails to resolve
it. Unlike the winding it is not whole by construction. Boxes outside `valid`
(optional boolean mask) are left out. Raises ValueError for a zero direction, radius
<= 0 or no box inside.
"""
from __future__ import annotations

import numpy as np


def whirl_strength(field: np.ndarray, centre, axis, radius: float, half_height: float,
                   valid: np.ndarray | None = None) -> float:
    """Angular momentum about the line through centre along axis, per unit density, nearby boxes."""
    field = np.asarray(field, dtype=np.complex128)
    direction = np.asarray(axis, dtype=np.float64)
    if not np.linalg.norm(direction) > 0 or radius <= 0:
        raise ValueError(f"need a nonzero axis and radius > 0, got {axis} and {radius}")
    direction = direction / np.linalg.norm(direction)
    places = np.stack(np.meshgrid(*[np.arange(size, dtype=np.float64) for size in field.shape], indexing="ij"),
                      axis=-1) - np.asarray(centre, dtype=np.float64)
    along = places @ direction
    across = places - along[..., None] * direction
    inside = (np.linalg.norm(across, axis=-1) <= radius) & (np.abs(along) <= half_height)
    if valid is not None:
        inside &= np.asarray(valid, dtype=bool)
    if not inside.any():
        raise ValueError("no box lies inside the cylinder round the line")
    gradient = np.stack(np.gradient(field), axis=-1)
    swirl = np.cross(places, gradient) @ direction
    current = np.imag(np.conj(field) * swirl)
    return float(current[inside].sum() / (np.abs(field[inside]) ** 2).sum())
