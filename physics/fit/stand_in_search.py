"""Stand-in search: random search on the stand-in score, many settings points side by side in one GPU job.

Contract: `stand_in_search(start, points, steps=24, every=3, batch=32, block=10,
device="cuda")` scores every row of `points` (N x 6, the fit's unit cube,
physics.fit.fit_search NAMES) with physics.fit.stand_in_score, `batch` universes side by
side per GPU run, each its own settings, all from `start` (one universe's signed quanta).
Returns {"score" (N), "parts" (N x 5), "seconds" (wall time of the whole search, every
batch synchronized)}. No gradient is taken. Raises ValueError when points is not (N, 6)
or batch < 1.
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

from physics.fit.fit_search import NAMES
from physics.fit.stand_in_score import stand_in_score


def stand_in_search(start: np.ndarray, points, steps: int = 24, every: int = 3, batch: int = 32, block: int = 10,
                    device: str = "cuda") -> dict:
    """The stand-in score of every point, `batch` universes side by side per run."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != len(NAMES) or batch < 1:
        raise ValueError(f"need points of shape (N, {len(NAMES)}) and batch >= 1, got {points.shape}, {batch}")
    started = time.perf_counter()
    scores, parts = [], []
    for first in range(0, len(points), batch):
        score, part = stand_in_score(start, wp.array(points[first:first + batch], device=device), steps, every,
                                     block, device)
        scores.append(score.numpy())
        parts.append(part.numpy())
    wp.synchronize_device(device)
    return {"score": np.concatenate(scores).astype(np.float64), "parts": np.concatenate(parts).astype(np.float64),
            "seconds": time.perf_counter() - started}
