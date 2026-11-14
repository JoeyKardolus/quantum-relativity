"""Validate (a): spatial dimension measured from the network alone.

Contract: `measure_dimensions(seed, out_dir)` builds spatial reference networks
of true dimension 1..4 (k-nearest connecting, coordinates discarded), measures
ball-growth and spectral dimension, plots measured vs true, and returns
(numbers, plot names, settings). A number passes when within TOLERANCE.
"""
from __future__ import annotations

import numpy as np

from physics.measure import ball_dimension, spectral_dimension
from physics.network import spatial_reference
from physics.experiments.validate._shared import new_figure, save, within

SIZES = {1: 10_000, 2: 20_000, 3: 30_000, 4: 40_000}


def measure_dimensions(seed: int, out_dir: str) -> tuple[list[dict], list[str], dict]:
    """Ball and spectral dimension of reference networks of known dimension 1..4."""
    numbers, ball, spectral = [], [], []
    for dimension, joints in SIZES.items():
        network, known = spatial_reference(joints, dimension, seed=seed)
        ball_value = ball_dimension(network, seed=seed)["dimension"]
        spectral_value = spectral_dimension(network, seed=seed)["dimension"]
        ball.append(ball_value)
        spectral.append(spectral_value)
        for label, value in (("ball-growth", ball_value), ("spectral", spectral_value)):
            numbers.append({"name": f"{label} dimension, {dimension}D space ({joints} joints)",
                            "measured": round(value, 3), "known": dimension,
                            "note": "within 10%" if within(value, dimension) else "MISS: outside 10%"})
    plot = _plot(list(SIZES), ball, spectral, out_dir)
    settings = {"sizes": SIZES, "neighbours": "default 2d+4", "ball": "sources=64, inner=3, outer=0.4",
                "spectral": "sources=64, steps=200, window=(20,200)"}
    return numbers, [plot], settings


def _plot(true: list[int], ball: list[float], spectral: list[float], out_dir: str) -> str:
    figure, axes = new_figure()
    span = np.array([0.5, 4.5])
    axes.plot(span, span, "--", color="grey", label="known: measured = true")
    axes.fill_between(span, span * 0.9, span * 1.1, color="grey", alpha=0.15, label="10% band")
    axes.plot(true, ball, "o", ms=8, label="ball growth N(r) ~ r^D")
    axes.plot(true, spectral, "s", ms=8, label="spectral (random-walk return)")
    axes.set(xlabel="true dimension of the space the network was built from",
             ylabel="dimension measured from the network",
             title="Does the network know its own dimension?", xticks=true)
    axes.legend()
    return save(figure, out_dir, "dimension.png")
