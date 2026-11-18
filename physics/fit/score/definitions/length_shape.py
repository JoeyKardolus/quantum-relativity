"""Length shape: how well one length's history matches the known expansion, slowing and then speeding up.

Contract: `length_shape(tau, lengths)` takes the observer's time at every sample and one
length at every sample (the universe's size, or the gap between its lumps) and returns
`{"shape", "growth", "dip"}`:
  growth  end over start, minus 1
  grew    gain(growth, GROWTH_SCALE): 0 when it did not grow
  steady  (end - start) / the sum of |change| from sample to sample: 1 for growth that
          never turns back, lower the more it shrinks in between, 0 when nothing moved
  dip     min(v1 - v2, v3 - v2) / mean |v|, with v1, v2, v3 the length gained in the
          three thirds of the observer's time: above 0 only when the growth slows first
          and then speeds up, the known history (the speed of growing, not the rate
          ln a, which in the real universe only ever falls)
  shape   grew * steady * (1 + dip clipped to -1 .. 1) / 2, between 0 and 1
Raises ValueError when the first length is not above 0.
"""
from __future__ import annotations

import numpy as np

from physics.fit.score.definitions.gain import gain
from physics.fit.score.definitions.score_marks import GROWTH_SCALE


def length_shape(tau: np.ndarray, lengths: list) -> dict:
    """grew * steady * (1 + dip) / 2 for one length over the observer's time, with its growth and dip."""
    scale = np.asarray(lengths, dtype=np.float64)
    if not scale[0] > 0.0:
        raise ValueError(f"a length must start above 0, got {scale[0]}")
    growth = float(scale[-1] / scale[0] - 1.0)
    speeds = np.diff(np.interp(np.linspace(tau[0], tau[-1], 4), tau, scale))
    dip = float(min(speeds[0] - speeds[1], speeds[2] - speeds[1]) / (np.abs(speeds).mean() + 1e-300))
    moved = float(np.abs(np.diff(scale)).sum())
    steady = max(float(scale[-1] - scale[0]), 0.0) / moved if moved > 0.0 else 0.0
    shape = gain(growth, GROWTH_SCALE) * steady * (1.0 + min(max(dip, -1.0), 1.0)) / 2.0
    return {"shape": shape, "growth": growth, "dip": dip}
