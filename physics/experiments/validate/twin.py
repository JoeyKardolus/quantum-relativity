"""Validate (c): the twin paradox as longest chains in a 1+1 causal network.

Contract: `measure_twin_paradox(seed, out_dir)` sprinkles a 1+1 diamond from
event A (bottom tip) to event B (top tip, proper time T) with detour events C at
(T/2, x) for several x. Stay-at-home twin: longest chain A -> B. Travelling twin:
longest chain A -> C plus C -> B. Flat-spacetime prediction for the traveller /
stay-at-home ratio: 2 * sqrt((T/2)^2 - x^2) / T. Ratios are compared rather than
raw lengths so the shared finite-size undershoot cancels to first order.
Returns (numbers, plot names, settings, pass flag).
"""
from __future__ import annotations

import numpy as np

from physics.measure import longest_chain
from physics.network import causal_reference
from physics.experiments.validate._shared import new_figure, save, within

JOINTS = 4000
SEEDS = 3
PROPER_TIME = 1.0
OFFSETS = (0.1, 0.2, 0.3, 0.4)


def measure_twin_paradox(seed: int, out_dir: str) -> tuple[list[dict], list[str], dict, bool]:
    """Traveller / stay-at-home longest-chain ratio vs the flat prediction."""
    anchors = np.array([[PROPER_TIME / 2, offset] for offset in OFFSETS])
    ratios = np.zeros((SEEDS, len(OFFSETS)))
    for row in range(SEEDS):
        network, known = causal_reference(JOINTS, 2, PROPER_TIME, anchors=anchors, seed=seed + row)
        home = longest_chain(network, known["bottom"], known["top"])
        for column, detour in enumerate(known["anchors"]):
            away = (longest_chain(network, known["bottom"], detour)
                    + longest_chain(network, detour, known["top"]))
            ratios[row, column] = away / home
    measured = ratios.mean(axis=0)
    predicted = np.array([2 * np.sqrt((PROPER_TIME / 2) ** 2 - offset ** 2) / PROPER_TIME
                          for offset in OFFSETS])
    shorter = bool(np.all(ratios < 1))
    passed = shorter and all(within(value, expected) for value, expected in zip(measured, predicted))
    numbers = [{"name": f"traveller / stay-at-home chain, detour x={offset} (speed {2 * offset:.1f}c)",
                "measured": round(float(value), 3), "known": round(float(expected), 3),
                "note": f"mean of {SEEDS} seeds, spread {ratios[:, column].std():.3f}"}
               for column, (offset, value, expected) in enumerate(zip(OFFSETS, measured, predicted))]
    numbers.append({"name": "traveller always has the shorter chain", "measured": shorter,
                    "known": True, "note": f"{ratios.size} trips checked"})
    plot = _plot(np.array(OFFSETS), ratios, predicted, out_dir)
    settings = {"joints": JOINTS, "seeds": SEEDS, "proper_time": PROPER_TIME, "detour_x": OFFSETS}
    return numbers, [plot], settings, passed


def _plot(offsets, ratios, predicted, out_dir: str) -> str:
    figure, axes = new_figure()
    smooth = np.linspace(0, 0.45, 100)
    axes.plot(smooth, 2 * np.sqrt(0.25 - smooth ** 2), "--", color="grey",
              label="known (flat spacetime): 2 sqrt(1/4 - x^2)")
    axes.fill_between(smooth, 1.8 * np.sqrt(0.25 - smooth ** 2), 2.2 * np.sqrt(0.25 - smooth ** 2),
                      color="grey", alpha=0.15, label="10% band")
    for row in ratios:
        axes.plot(offsets, row, ".", color="tab:blue", alpha=0.4)
    axes.plot(offsets, ratios.mean(axis=0), "o", ms=8, color="tab:blue", label="measured (mean of seeds)")
    axes.axhline(1.0, color="black", lw=0.8, label="stay-at-home twin")
    axes.set(xlabel="how far the traveller goes out (x of turn-around event, T = 1)",
             ylabel="traveller's chain / stay-at-home chain",
             title="Twin paradox: the traveller ages less", ylim=(0, 1.1))
    axes.legend(loc="lower left")
    return save(figure, out_dir, "twin_paradox.png")
