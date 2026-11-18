"""In plane: the two axes a unit square of boxes lies in, and the way it is walked.

Contract: `in_plane(normal)` takes a box axis (0, 1 or 2) and returns `(across, up)`,
the two other axes in the order `(normal + 1) % 3, (normal + 2) % 3`. A square with that
normal is walked +across, +up, -across, -up: right-handed about +normal. Every winding
in this package has its sign from this one rule. The axes are those of a cell's own
block of boxes; they are no directions in space. Raises ValueError for any other normal.
"""
from __future__ import annotations

AXES = 3


def in_plane(normal: int) -> tuple[int, int]:
    """The two axes across `normal`, in right-handed order."""
    if normal not in range(AXES):
        raise ValueError(f"normal must be 0, 1 or 2, got {normal}")
    return (normal + 1) % AXES, (normal + 2) % AXES
