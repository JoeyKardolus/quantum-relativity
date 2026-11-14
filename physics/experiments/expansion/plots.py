"""PNG curves for the expansion test.

Contract: `draw_plots(runs, drift, out_dir)` takes the per-run summaries (each
with its `evolve` record) and writes three PNGs into out_dir (scale factor,
growth rate, cluster size), each with the known drift-only value dashed.
Returns the file names.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def draw_plots(runs: list[dict], drift: float, out_dir: str) -> list[str]:
    """Write the four figures; return their file names."""
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))
    return [_scale_plot(runs, drift, colours, out_dir), _rate_plot(runs, drift, colours, out_dir),
            _cluster_plot(runs, colours, out_dir)]


def _label(run: dict) -> str:
    if not run["attraction_multiple"]:
        return "drift only"
    tail = f", broke down at step {run['record']['breakdown_step']}" if run["record"]["breakdown_step"] else ""
    return (f"pull x{run['attraction_multiple']:g}, limit x{run['limit_multiple']:g}, "
            f"push {run['joint_push']:g}{tail}")


def _save(figure, out_dir: str, name: str) -> str:
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, name), dpi=110)
    plt.close(figure)
    return name


def _scale_plot(runs, drift, colours, out_dir) -> str:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for run, colour in zip(runs, colours):
        steps, scale = np.array(run["record"]["step"]), np.array(run["record"]["scale"])
        axis.plot(steps, scale / scale[0], color=colour, label=_label(run))
    longest = max(run["record"]["step"][-1] for run in runs)
    grid = np.arange(longest + 1)
    axis.plot(grid, (1 + drift) ** grid, "k--", label=f"pure exponential (1+h)^t, h = {drift}")
    axis.set_yscale("log")
    axis.set_xlabel("step")
    axis.set_ylabel("scale factor a(t)/a(0)  (sparse-space path distance)")
    axis.set_title("Sparse space: distance between far-apart joints")
    axis.legend(fontsize=7)
    return _save(figure, out_dir, "scale_factor.png")


def _rate_plot(runs, drift, colours, out_dir) -> str:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for run, colour in zip(runs, colours):
        axis.plot(run["record"]["step"], run["rate"], color=colour, marker=".", label=_label(run))
    axis.axhline(np.log(1 + drift), color="k", ls="--", label=f"drift only: ln(1+h) = {np.log(1 + drift):.4f}")
    axis.set_xlabel("step")
    axis.set_ylabel("growth rate H = d ln a / dt  (per step)")
    axis.set_title("Expansion rate: rising = accelerating faster than exponential")
    axis.legend(fontsize=7)
    return _save(figure, out_dir, "growth_rate.png")


def _cluster_plot(runs, colours, out_dir) -> str:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for run, colour in zip(runs, colours):
        steps = run["record"]["step"]
        sizes = np.array(run["record"]["cluster_size"])
        scale = np.array(run["record"]["scale"])
        axis.plot(steps, (sizes / sizes[0]).mean(axis=1), color=colour, label=_label(run))
        axis.plot(steps, scale / scale[0], color=colour, ls=":", lw=1)
    axis.axhline(1.0, color="k", ls="--", label="bound: size stays 1")
    axis.set_yscale("log")
    axis.set_xlabel("step")
    axis.set_ylabel("cluster internal size / start  (solid)\nscale factor (dotted)")
    axis.set_title("Dense clusters: do they grow with space or stay bound?")
    axis.legend(fontsize=7)
    return _save(figure, out_dir, "cluster_size.png")
