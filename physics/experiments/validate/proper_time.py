"""Validate (b): proper time as the longest causal chain in flat 1+1 spacetime.

Contract: `measure_proper_time(seed, out_dir)` sprinkles 1+1 causal diamonds of
proper time T at several sizes (so several densities), takes the longest chain
bottom -> top, compares with sqrt(2 * density) * T, plots it, and returns
(numbers, plot names, settings). Known: the ratio approaches 1 from below as
the size grows; pass = largest size within TOLERANCE and the ratio rising with size.
"""
from __future__ import annotations

import numpy as np

from physics.measure import longest_chain
from physics.network import causal_reference
from physics.experiments.validate._shared import new_figure, save, within

SIZES = (250, 500, 1000, 2000, 4000)
SEEDS = 10
PROPER_TIME = 1.0


def measure_proper_time(seed: int, out_dir: str) -> tuple[list[dict], list[str], dict, bool]:
    """Longest chain vs sqrt(2*density)*T over sizes; returns the pass flag too."""
    measured, predicted = [], []
    for joints in SIZES:
        chains = []
        for offset in range(SEEDS):
            network, known = causal_reference(joints, 2, PROPER_TIME, seed=seed + offset)
            chains.append(longest_chain(network, known["bottom"], known["top"]))
        measured.append(float(np.mean(chains)))
        predicted.append(float(np.sqrt(2 * known["density"]) * PROPER_TIME))
    ratios = np.array(measured) / np.array(predicted)
    rising = bool(np.all(np.diff(ratios) > -0.02))
    passed = within(measured[-1], predicted[-1]) and rising
    numbers = [{"name": f"longest chain, 1+1 diamond, {joints} joints (density {joints / 0.5:.0f})",
                "measured": round(value, 2), "known": round(expected, 2),
                "note": f"ratio {value / expected:.3f}; mean of {SEEDS} seeds"}
               for joints, value, expected in zip(SIZES, measured, predicted)]
    numbers.append({"name": "longest chain ratio rises toward 1 with size", "measured": rising,
                    "known": True, "note": f"ratios {[round(float(r), 3) for r in ratios]}"})
    plot = _plot(np.array(SIZES), np.array(measured), np.array(predicted), ratios, out_dir)
    settings = {"sizes": SIZES, "seeds_per_size": SEEDS, "proper_time": PROPER_TIME}
    return numbers, [plot], settings, passed


def _plot(sizes, measured, predicted, ratios, out_dir: str) -> str:
    figure, (left, right) = new_figure(11, 4.5, panels=2)
    left.plot(predicted, predicted, "--", color="grey", label="known: sqrt(2 density) T")
    left.plot(predicted, measured, "o-", label="longest chain (steps)")
    left.set(xlabel="flat-spacetime prediction sqrt(2 density) T", ylabel="longest chain length",
             title="Proper time = longest causal chain")
    left.legend()
    right.axhline(1.0, ls="--", color="grey", label="known ratio 1 (infinite size)")
    right.axhspan(0.9, 1.1, color="grey", alpha=0.15, label="10% band")
    right.semilogx(sizes, ratios, "o-", label="measured / predicted")
    right.set_xticks(sizes, [str(size) for size in sizes], minor=False)
    right.minorticks_off()
    right.set(xlabel="joints in the diamond", ylabel="ratio",
              title="Approach to the known value with size")
    right.legend()
    return save(figure, out_dir, "proper_time.png")
