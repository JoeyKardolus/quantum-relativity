"""Went round: how far a blob's centre went round the big volume, and Kepler's ratio for it.

Contract: `went_round(times, extra, centres)` takes per sample the time, the blob's
quanta at its own volume and its centre (three floats from the big volume, blob_centre).
Returns a dict:
  lost_sample   the first sample at which `extra` is under half its first value (-1: never)
  radius        the centre's distance from the big volume per sample, in volumes
  turned        the angle the centre went round in the plane of the first two axes, in
                whole turns (signed), summed over the samples up to lost_sample
  period_squared_over_radius_cubed   (time taken / |turned|) ** 2 / radius[0] ** 3 from
                that angle: Kepler's ratio; 0.0 when it did not turn or started at the middle
Raises ValueError for no samples or lists of unequal length.
"""
from __future__ import annotations

import math

import numpy as np


def went_round(times, extra, centres) -> dict:
    """When the blob was lost, its radius per sample, the turns it made and period squared over radius cubed."""
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    if not len(times) or not len(times) == len(extra) == len(centres):
        raise ValueError("need one time, one amount and one centre per sample")
    lost = next((index for index, amount in enumerate(extra) if amount < 0.5 * extra[0]), -1)
    kept = centres if lost < 0 else centres[:lost + 1]
    angle = np.unwrap(np.arctan2(kept[:, 1], kept[:, 0]))
    turned = float(angle[-1] - angle[0]) / (2.0 * math.pi)
    radius = np.linalg.norm(centres, axis=1)
    kepler = 0.0
    if abs(turned) > 1e-9 and radius[0] > 0.0:
        kepler = float((times[len(kept) - 1] / abs(turned)) ** 2 / radius[0] ** 3)
    return {"lost_sample": lost, "radius": radius.tolist(), "turned": turned,
            "period_squared_over_radius_cubed": kepler}
