"""Shared bits for the Validate experiment: the pass band and the plot setup.

Contract: `TOLERANCE` is the relative band (decided before any run) a measured
number must land in to count as recovered; `within(measured, known)` applies it;
`new_figure()` and `save(figure, out_dir, name)` give Agg-backend PNG plots.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOLERANCE = 0.10


def within(measured: float, known: float) -> bool:
    """True when measured lies within TOLERANCE (relative) of known."""
    return abs(measured - known) <= TOLERANCE * abs(known)


def new_figure(width: float = 7.0, height: float = 4.5, panels: int = 1):
    """A fresh figure and its axes (a tuple when panels > 1)."""
    return plt.subplots(1, panels, figsize=(width, height), dpi=110)


def save(figure, out_dir: str, name: str) -> str:
    """Write the figure as out_dir/name, close it, return the file name."""
    os.makedirs(out_dir, exist_ok=True)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, name))
    plt.close(figure)
    return name
