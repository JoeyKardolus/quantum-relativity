"""One PNG of several curves over time.

Contract: `plot_curves(path, title, xlabel, ylabel, curves, known)` draws each
curve (label -> (x, y)) with matplotlib's Agg backend, a dashed horizontal line
at `known` when it is not None (log y axis when `log_y`), and writes the PNG to `path`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_curves(path: str, title: str, xlabel: str, ylabel: str,
                curves: dict[str, tuple], known: float | None = None,
                known_label: str = "known", log_y: bool = False) -> None:
    """Write a line plot of the given curves to `path`."""
    figure, axes = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    for label, (xs, ys) in curves.items():
        axes.plot(xs, ys, label=label, linewidth=1.8)
    if known is not None:
        axes.axhline(known, linestyle="--", color="grey", label=known_label)
    if log_y:
        axes.set_yscale("log")
    axes.set_title(title, fontsize=11)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
