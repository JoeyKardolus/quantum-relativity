"""fit_score: the fit's score of one universe's samples, 0 when nothing happens and at most 1.

Contract: fit_score(samples) -> dict. `samples` are fit_measure dicts in time order, each
with "tau", the observer's time at the sample (fit_run); the first is the start.
score = 0.5 E + 0.5 S, both between 0 and 1, every part a change since the start, so a
universe where nothing happens scores exactly 0, the worst (owner: an honest score).
  E  expansion shape. a = cube root of `room` (space between lumps, in boxes). growth =
     a_end / a_start - 1; G = tanh(growth / 0.06), 0 when space between lumps did not
     grow. The run is cut in thirds of the observer's time and r1, r2, r3 are the
     growth rates of ln a in them; dip = min(r1 - r2, r3 - r2) / mean |r|, above 0 only
     when growth slows first and then speeds up (the known history). E = G * (1 + dip
     clipped to -1 .. 1) / 2, and 0 when space shrank in any third.
  S  structure, the mean of four parts:
     Sr  tanh(gain of region_contrast / 1.0): unevenness over regions grew
     Sl  tanh(gain of lump_share / 0.4): more of the energy sits in lumps
     Ss  tanh(sealed_share at the end / 0.1): lumps sealed themselves
     Sp  tanh(gain of lump_spread / 0.3): the lump volumes' sizes spread out (needs 3)
     each 0 for a loss.
The scales (0.06, 1.0, 0.4, 0.1, 0.3) were fixed on 2026-11-17 after one look at the
default universe and before the first search with this score; they say only where a
part stops counting more and are not fitted.
Raises ValueError for fewer than 4 samples or an observer's time that does not advance.
"""
from __future__ import annotations

import math

import numpy as np


def fit_score(samples: list) -> dict:
    """Score, its expansion and structure halves and every part, from a run's samples."""
    if len(samples) < 4:
        raise ValueError(f"need at least 4 samples, got {len(samples)}")
    tau = np.array([sample["tau"] for sample in samples], dtype=np.float64)
    if not np.all(np.diff(tau) > 0.0):
        raise ValueError("the observer's time must advance between samples")
    scale = np.cbrt(np.array([sample["room"] for sample in samples], dtype=np.float64))
    growth = float(scale[-1] / scale[0] - 1.0)
    cuts = np.interp(np.linspace(tau[0], tau[-1], 4), tau, np.log(scale))
    rates = np.diff(cuts) / ((tau[-1] - tau[0]) / 3.0)
    dip = float(min(rates[0] - rates[1], rates[2] - rates[1]) / (np.abs(rates).mean() + 1e-30))
    grew = math.tanh(max(growth, 0.0) / 0.06)
    expansion = 0.0 if rates.min() <= 0.0 else grew * (1.0 + min(max(dip, -1.0), 1.0)) / 2.0
    first, last = samples[0], samples[-1]
    parts = {"Sr": _gain(last["region_contrast"] - first["region_contrast"], 1.0),
             "Sl": _gain(last["lump_share"] - first["lump_share"], 0.4),
             "Ss": _gain(last["sealed_share"], 0.1),
             "Sp": _gain(last["lump_spread"] - first["lump_spread"], 0.3) if last["lump_volumes"] >= 3 else 0.0}
    structure = sum(parts.values()) / 4.0
    return {"score": 0.5 * expansion + 0.5 * structure, "E": expansion, "S": structure, "growth": growth, "G": grew,
            "dip": dip, "r1": float(rates[0]), "r2": float(rates[1]), "r3": float(rates[2]), **parts}


def _gain(change: float, scale: float) -> float:
    """tanh(change / scale), 0 for a loss."""
    return math.tanh(max(change, 0.0) / scale)
