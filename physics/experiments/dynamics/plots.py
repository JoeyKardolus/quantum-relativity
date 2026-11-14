"""Drift plot for Grid dynamics.

Contract: `plot_drift(seed, out_dir)` steps a 3D reference network 50 times
under drift alone and under drift with energy flow, writes one PNG of mean
connection length against (1 + h)^n, and returns its file name.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physics.network import spatial_reference  # noqa: E402
from physics.rules import RuleSettings, step  # noqa: E402

_STEPS = 50
_DRIFT = 0.01


def plot_drift(seed: int, out_dir: str) -> str:
    """Write drift.png; return its file name."""
    base = spatial_reference(2000, 3, seed=seed)[0]
    figure, axes = plt.subplots(figsize=(6, 4))
    for label, settings in (("drift only", RuleSettings(drift=_DRIFT)),
                            ("drift + energy flow", RuleSettings(drift=_DRIFT, energy_flow=0.1))):
        axes.plot(range(_STEPS + 1), _mean_lengths(base, settings) / base.length.mean(), label=label)
    steps = np.arange(_STEPS + 1)
    axes.plot(steps, (1 + _DRIFT) ** steps, "k--", label="known (1 + h)^n")
    axes.set_xlabel("step")
    axes.set_ylabel("mean connection length / start")
    axes.set_title(f"Drift h = {_DRIFT}, 2000 joints, seed {seed}")
    axes.legend()
    figure.tight_layout()
    name = "drift.png"
    figure.savefig(os.path.join(out_dir, name), dpi=110)
    plt.close(figure)
    return name


def _mean_lengths(base, settings) -> np.ndarray:
    means, network = [base.length.mean()], base
    for _ in range(_STEPS):
        network = step(network, settings)
        means.append(network.length.mean())
    return np.array(means)
