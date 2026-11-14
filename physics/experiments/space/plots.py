"""PNG plots for the space test.

Contract: `plot_tracks(tracks, out_dir)` takes {start name: track history} and
writes three PNGs into out_dir: ball-growth dimension vs step, spectral
dimension vs step (both with 3D space as a dashed line; a missing point is a
step where the measurement had no valid range (drawn as an x at the bottom)), and steps across the network vs
step. Returns the file names.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
_GAP_NOTE = "x along the bottom: no valid fit range (small-world, or too few joints)"


def plot_tracks(tracks: dict, out_dir: str) -> list[str]:
    """Write the three space-test plots and return their file names."""
    os.makedirs(out_dir, exist_ok=True)
    names = [
        _dimension_plot(tracks, "ball", "Ball-growth dimension while the rules run",
                        "ball-growth dimension D (N(r) ~ r^D)", out_dir, "space_ball_dimension.png"),
        _dimension_plot(tracks, "spectral", "Spectral dimension while the rules run",
                        "spectral dimension ds (P(t) ~ t^(-ds/2))", out_dir,
                        "space_spectral_dimension.png"),
        _across_plot(tracks, out_dir, "space_steps_across.png"),
    ]
    return names


def _series(values: list) -> list[float]:
    return [float("nan") if value is None else value for value in values]


def _dimension_plot(tracks: dict, key: str, title: str, label: str, out_dir: str,
                    name: str) -> str:
    figure, axes = plt.subplots(figsize=(8, 5))
    order = 0
    for colour, (start, history) in zip(_COLOURS, tracks.items()):
        axes.plot(history["step"], _series(history[key]), "-o", color=colour, lw=2, ms=5,
                  label=start)
        lost = [at for at, value in zip(history["step"], history[key]) if value is None]
        axes.plot(lost, [0.25 + 0.2 * order] * len(lost), "x", color=colour, ms=6)
        order += 1
    axes.axhline(3, ls="--", color="#52514e", lw=1.2, label="3D space (target)")
    axes.set(title=title, xlabel="rule step (log scale; 0 = start)", ylabel=label, ylim=(0, 7),
             xscale="symlog", xlim=(-0.1, None))
    axes.text(0.01, 0.97, _GAP_NOTE, transform=axes.transAxes, fontsize=8, color="#52514e", va="top")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=9, loc="upper right")
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, name), dpi=120)
    plt.close(figure)
    return name


def _across_plot(tracks: dict, out_dir: str, name: str) -> str:
    figure, axes = plt.subplots(figsize=(8, 5))
    for colour, (start, history) in zip(_COLOURS, tracks.items()):
        axes.plot(history["step"], history["across"], "-o", color=colour, lw=2, ms=5,
                  label=start)
    axes.set(title="How many connections it takes to cross the network",
             xlabel="rule step (log scale; 0 = start)",
             ylabel="mean largest distance / median connection length", xscale="symlog", xlim=(-0.1, None),
             yscale="log")
    axes.grid(alpha=0.25, which="both")
    axes.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, name), dpi=120)
    plt.close(figure)
    return name
