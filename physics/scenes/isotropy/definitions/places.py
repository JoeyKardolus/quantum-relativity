"""Places: where round a lump the isotropy scene compares on-axis with off-axis.

Contract: `PAIRS` holds pairs of places (i, j, k), in volumes from the lump, at the same
straight-line distance, the first on a lattice axis and the second off it: 3, 5, 6 and
7 volumes out. `DIAGONALS` holds three places the same count of cells from the lump
along an axis, a face diagonal and a body diagonal (their straight-line distances
differ; reported, no pass mark). Written 2026-11-17 before any run.
"""
from __future__ import annotations

PAIRS = (((3, 0, 0), (2, 2, 1)), ((5, 0, 0), (4, 3, 0)), ((6, 0, 0), (4, 4, 2)), ((7, 0, 0), (6, 3, 2)))
DIAGONALS = ((3, 0, 0), (3, 3, 0), (3, 3, 3))
