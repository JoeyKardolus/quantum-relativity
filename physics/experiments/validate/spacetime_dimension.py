"""Validate (d): spacetime dimension from the causal order (Myrheim-Meyer).

Contract: `measure_spacetime_dimension(seed, out_dir)` sprinkles 1+1 and 3+1
causal diamonds, measures the Myrheim-Meyer dimension from the ordering
fraction only, compares with the true 2 and 4, plots it, and returns
(numbers, plot names, settings).
"""
from __future__ import annotations

import numpy as np

from physics.measure import myrheim_meyer_dimension
from physics.network import causal_reference
from physics.experiments.validate._shared import new_figure, save, within

JOINTS = 2000
SEEDS = 3


def measure_spacetime_dimension(seed: int, out_dir: str) -> tuple[list[dict], list[str], dict]:
    """Myrheim-Meyer dimension of 1+1 and 3+1 diamonds vs the true 2 and 4."""
    numbers, results = [], {}
    for dimension in (2, 4):
        values = [myrheim_meyer_dimension(causal_reference(JOINTS, dimension, seed=seed + offset)[0])
                  ["dimension"] for offset in range(SEEDS)]
        results[dimension] = values
        mean = float(np.mean(values))
        numbers.append({"name": f"Myrheim-Meyer dimension, {dimension - 1}+1 diamond ({JOINTS} joints)",
                        "measured": round(mean, 3), "known": dimension,
                        "note": f"mean of {SEEDS} seeds, spread {np.std(values):.3f}; "
                                + ("within 10%" if within(mean, dimension) else "MISS: outside 10%")})
    plot = _plot(results, out_dir)
    return numbers, [plot], {"joints": JOINTS, "seeds": SEEDS}


def _plot(results: dict, out_dir: str) -> str:
    figure, axes = new_figure()
    span = np.array([1.5, 4.5])
    axes.plot(span, span, "--", color="grey", label="known: measured = true")
    axes.fill_between(span, span * 0.9, span * 1.1, color="grey", alpha=0.15, label="10% band")
    for dimension, values in results.items():
        axes.plot([dimension] * len(values), values, "o", ms=8, color="tab:purple", alpha=0.6,
                  label="measured (one dot per seed)" if dimension == 2 else None)
    axes.set(xlabel="true spacetime dimension (1+1 = 2, 3+1 = 4)",
             ylabel="Myrheim-Meyer dimension from causal order only",
             title="Spacetime dimension from 'who can influence whom'", xticks=[2, 4])
    axes.legend()
    return save(figure, out_dir, "spacetime_dimension.png")
