"""The expansion test's one picture.

Contract: `draw_picture(runs, out_dir)` writes one PNG with two panels and
returns its file name. Left, the runs at the highest push limit: sparse-space
scale factor (solid) and mean cluster size (dashed), each over its start, one
colour per drift excess, drawn down to FLOOR; a run that fell in gets an x at
its breakdown step. Right, every run at the lowest (house) push limit:
the share of sparse joints not yet fallen into a collapsed region, per step,
with the v2 runaway marked.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAME = "expansion.png"
V2_RUNAWAY = (4, 184 / 5600)   # v2 page, seed 1: step, share of joints left
FLOOR = 0.5                    # left panel: lowest size drawn; a run that fell in is marked here


def draw_picture(runs: list[dict], out_dir: str) -> str:
    """Write the two-panel figure; return its file name."""
    excesses = sorted({entry["drift_excess"] for entry in runs})
    colours = dict(zip(excesses, plt.cm.viridis(np.linspace(0, 0.85, len(excesses)))))
    highest = max(entry["limit_multiple"] for entry in runs)
    lowest = min(entry["limit_multiple"] for entry in runs)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    _sizes(left, [entry for entry in runs if entry["limit_multiple"] == highest], colours, highest)
    _survivors(right, [entry for entry in runs if entry["limit_multiple"] == lowest], colours, lowest)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, NAME), dpi=110)
    plt.close(figure)
    return NAME


def _tag(entry: dict) -> str:
    return f"drift = balance {entry['drift_excess']:+.0%}"


def _sizes(axis, runs: list[dict], colours: dict, multiple: float) -> None:
    for entry in runs:
        steps = entry["record"]["step"]
        scale = np.array(entry["record"]["scale"])
        sizes = np.nanmean(np.array(entry["record"]["cluster_size"])
                           / np.array(entry["record"]["cluster_size"][0]), axis=1)
        colour = colours[entry["drift_excess"]]
        axis.plot(steps, scale / scale[0], color=colour, label=f"{_tag(entry)}: space")
        axis.plot(steps, sizes, color=colour, ls="--", label=f"{_tag(entry)}: clusters")
        if entry["record"]["breakdown_step"]:
            axis.plot(entry["record"]["breakdown_step"], FLOOR, "x", color=colour, ms=9)
    axis.axhline(1.0, color="k", lw=0.8, ls=":")
    axis.set_ylim(FLOOR - 0.05, 1.5)
    axis.set_xlabel("step")
    axis.set_ylabel("size / size at start (path distance); x = fell in")
    axis.set_title(f"Push limit {multiple:g}x background push: space (solid), clusters (dashed)",
                   fontsize=10)
    axis.legend(fontsize=7)


def _survivors(axis, runs: list[dict], colours: dict, multiple: float) -> None:
    for entry in runs:
        left = np.array(entry["record"]["sparse_by_step"])
        axis.plot(np.arange(len(left)), left / left[0], color=colours[entry["drift_excess"]],
                  marker=".", label=_tag(entry))
    axis.plot(*V2_RUNAWAY, "rx", ms=10, label="v2: all but 184 of 5600 joints gone at step 4")
    axis.set_xlabel("step")
    axis.set_ylabel("share of sparse joints still standing")
    axis.set_ylim(-0.05, 1.05)
    axis.set_title(f"Clusters past the push limit ({multiple:g}x, house value): does space survive?",
                   fontsize=10)
    axis.legend(fontsize=7)
