"""Gradient fit: climb the stand-in score by Warp's automatic gradients, several starts side by side.

Contract: `gradient_fit(start, starts, iterations=15, rate=0.05, steps=24, every=3, block=10,
device="cuda")` returns a list of one record per iteration: {"iteration", "units" (U x 6),
"score" (U), "parts" (U x 5), "gradient" (U x 6), "seconds"}, the score and gradient those
of the units at that iteration's start, then one more record with the last units' score
(gradient None). `start` is one universe's signed quanta (physics.engine.host_boxes);
`starts` (U x 6) the unit-cube points to climb from (physics.fit.fit_search NAMES), all U
climbed side by side in one GPU job. Each iteration runs physics.fit.stand_in_score under a
wp.Tape, takes d score / d units by tape.backward and moves each point by one Adam step
(beta 0.9, 0.999; the step's size is `rate` of the unit cube per setting, whatever the
gradient's size), clipped to [0, 1]. Adam is chosen because the gradient's size is
unreliable: the stand-in reads the box potentials, openness and clocks from a density the
tape does not see (stand_in_score), so the gradient misses the stiff feedback of a box's
own quanta on its phase, which grows without bound through a run (measured 2026-11-16: the
full gradient reached 1e26 on a 12^3 universe in 10 steps). It is a direction, not a
derivative. "seconds" is each iteration's wall time, forward and backward. Raises
ValueError when starts is not (U, 6) or iterations < 1.
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

from physics.fit.fit_search import NAMES
from physics.fit.stand_in_score import stand_in_score

_BETA_MEAN = 0.9
_BETA_SQUARE = 0.999


def gradient_fit(start: np.ndarray, starts, iterations: int = 15, rate: float = 0.05, steps: int = 24, every: int = 3,
                 block: int = 10, device: str = "cuda") -> list:
    """Adam ascent of the stand-in score from each start, gradients by Warp's tape."""
    units = np.asarray(starts, dtype=np.float64)
    if units.ndim != 2 or units.shape[1] != len(NAMES) or iterations < 1:
        raise ValueError(f"need starts of shape (U, {len(NAMES)}) and iterations >= 1, got {units.shape}, {iterations}")
    mean = np.zeros_like(units)
    square = np.zeros_like(units)
    records = []
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        score, parts, gradient = _score_and_gradient(start, units, steps, every, block, device)
        records.append({"iteration": iteration - 1, "units": units.tolist(), "score": score.tolist(),
                        "parts": parts.tolist(), "gradient": gradient.tolist(),
                        "seconds": time.perf_counter() - started})
        mean = _BETA_MEAN * mean + (1 - _BETA_MEAN) * gradient
        square = _BETA_SQUARE * square + (1 - _BETA_SQUARE) * gradient ** 2
        mean_hat = mean / (1 - _BETA_MEAN ** iteration)
        square_hat = square / (1 - _BETA_SQUARE ** iteration)
        units = np.clip(units + rate * mean_hat / (np.sqrt(square_hat) + 1e-12), 0.0, 1.0)
    started = time.perf_counter()
    last, last_parts = stand_in_score(start, wp.array(units.astype(np.float32), device=device), steps, every,
                                      block, device)
    records.append({"iteration": iterations, "units": units.tolist(), "score": last.numpy().tolist(),
                    "parts": last_parts.numpy().tolist(), "gradient": None,
                    "seconds": time.perf_counter() - started})
    return records


def _score_and_gradient(start, units, steps, every, block, device) -> tuple:
    """The stand-in score, its parts and d score / d units for every point, one taped run."""
    taped = wp.array(units.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        score, parts = stand_in_score(start, taped, steps, every, block, device)
    tape.backward(grads={score: wp.ones(score.shape, dtype=wp.float32, device=device)})
    result = (score.numpy().astype(np.float64), parts.numpy().astype(np.float64),
              taped.grad.numpy().astype(np.float64))
    tape.reset()
    return result
