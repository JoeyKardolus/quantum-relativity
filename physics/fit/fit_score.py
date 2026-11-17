"""fit_score: the fit's score of one universe's samples, 0 when nothing happens and at most 1.

Contract: fit_score(samples, held_share=0.0) -> dict. `samples` are fit_measure dicts in
time order, each with "tau", the observer's time at the sample, and "gap", the distance
between the end's lumps then (fit_gap; None when the run ends with fewer than two);
the first sample is the start. `held_share` is fit_gap's share of energy in lumps that
held. The definition, written down on 2026-11-17 before any search on the cell grid:

  score = hold * (E + S) / 2          every factor between 0 and 1

  E  expansion shape, the mean of the same number for two lengths: the size of the
     universe (the sum of its cell lengths) and the gap between its lumps (the summed
     lengths of the cells between them; 0 when there are no two lumps). For a length a:
       grew   = tanh(growth / 0.1), growth = a_end / a_start - 1, 0 when it did not grow
       steady = (a_end - a_start) / the sum of |change| from sample to sample: 1 for
                growth that never turns back, lower the more it shrinks in between
       dip    = min(v1 - v2, v3 - v2) / mean |v|, clipped to -1 .. 1, with v1, v2, v3
                the speeds da / dtau in the three thirds of the observer's time: above
                0 only when the growth slows first and then speeds up, the known
                history (the speed of growing, not the rate ln a, which in the real
                universe only ever falls)
       shape  = grew * steady * (1 + dip) / 2
  S  structure, the mean of five gains since the start, each 0 for a loss:
       Sr  tanh(gain of region_contrast / 1.0)   unevenness over regions grew
       Sl  tanh(gain of lump_share / 0.4)        more of the energy sits in lumps
       Sh  tanh((held_share - the start's lump_share) / 0.4)   lumps that hold: the
           energy in lumps two thirds through is still in lumps at the end
       Ss  tanh(the smaller of the gains of sealed_share and lump_share / 0.1)   lumps
           sealed themselves, counted only as far as energy gathered into lumps (a
           universe whose joints all seal at once freezes its start and gathered none)
       Sp  tanh(gain of lump_spread / 0.3) * Sl   the lump volumes' sizes spread out,
           counted only as far as energy gathered into lumps (needs 3; a stand-in
           for a web: no measure of threads is built)
  hold = alive * (1 - spent_share) at the end. alive is the share of the start's energy
     volumes still there, so a universe that fell into a single cell scores about 0;
     spent_share is the share of the size lying in overstretched volumes (past the tear
     length, or at the engine's floor, where a tear chain ends), so a universe that
     tears without end scores about 0, whatever its size did. Both fall off in a straight line: there is no cliff, and a collapse
     that makes one black hole costs that one volume's share.
Every part is a change since the start, so a universe where nothing happens scores
exactly 0; tanh caps every part, so no runaway earns more than steady growth does.
The scales (0.1, 1.0, 0.4, 0.4, 0.1, 0.3) say only where a part stops counting more;
they were fixed with this definition and are not fitted. Amended twice before the search
that counts, which each time started again from nothing. After 46 runs: Ss was the plain
gain of sealed_share (a universe frozen by sealing every joint at the start scored 0.10
with nothing gathered), and spent_share counted only volumes at the floor (a tear chain
caught midway kept a hold of 0.16). After 553 runs: the gap was the shortest chain by
length at every sample and Sp the plain gain of lump_spread; universes whose energy only
evened out (lump share 0.29 to 0.02) scored up to 0.16, because a route picked for its
short, full volumes lengthens as they empty, and the few lump volumes left lie further
apart in size.
Raises ValueError for fewer than 4 samples or an observer's time that does not advance.
"""
from __future__ import annotations

import math

import numpy as np


def fit_score(samples: list, held_share: float = 0.0) -> dict:
    """Score, its expansion and structure halves, the hold factor and every part, from a run's samples."""
    if len(samples) < 4:
        raise ValueError(f"need at least 4 samples, got {len(samples)}")
    tau = np.array([sample["tau"] for sample in samples], dtype=np.float64)
    if not np.all(np.diff(tau) > 0.0):
        raise ValueError("the observer's time must advance between samples")
    size = _shape(tau, [sample["size"] for sample in samples])
    gaps = [sample.get("gap") for sample in samples]
    gap = _shape(tau, gaps) if all(value is not None for value in gaps) else {"shape": 0.0, "growth": 0.0, "dip": 0.0}
    first, last = samples[0], samples[-1]
    parts = {"Sr": _gain(last["region_contrast"] - first["region_contrast"], 1.0),
             "Sl": _gain(last["lump_share"] - first["lump_share"], 0.4),
             "Sh": _gain(held_share - first["lump_share"], 0.4),
             "Ss": _gain(min(last["sealed_share"] - first["sealed_share"], last["lump_share"] - first["lump_share"]), 0.1),
             "Sp": _gain(last["lump_spread"] - first["lump_spread"], 0.3) if last["lump_volumes"] >= 3 else 0.0}
    parts["Sp"] *= parts["Sl"]
    expansion, structure = 0.5 * (size["shape"] + gap["shape"]), sum(parts.values()) / 5.0
    hold = min(max(last["alive"], 0.0), 1.0) * (1.0 - min(max(last["spent_share"], 0.0), 1.0))
    return {"score": hold * 0.5 * (expansion + structure), "E": expansion, "S": structure, "hold": hold,
            "E_size": size["shape"], "E_gap": gap["shape"], "growth": size["growth"], "dip": size["dip"],
            "gap_growth": gap["growth"], "gap_dip": gap["dip"], **parts}


def _shape(tau: np.ndarray, lengths: list) -> dict:
    """grew * steady * (1 + dip) / 2 for one length over the observer's time, with its growth and dip."""
    scale = np.asarray(lengths, dtype=np.float64)
    growth = float(scale[-1] / scale[0] - 1.0)
    cuts = np.interp(np.linspace(tau[0], tau[-1], 4), tau, scale)
    speeds = np.diff(cuts)
    dip = float(min(speeds[0] - speeds[1], speeds[2] - speeds[1]) / (np.abs(speeds).mean() + 1e-300))
    moved = float(np.abs(np.diff(scale)).sum())
    steady = max(float(scale[-1] - scale[0]), 0.0) / moved if moved > 0.0 else 0.0
    shape = math.tanh(max(growth, 0.0) / 0.1) * steady * (1.0 + min(max(dip, -1.0), 1.0)) / 2.0
    return {"shape": shape, "growth": growth, "dip": dip}


def _gain(change: float, scale: float) -> float:
    """tanh(change / scale), 0 for a loss."""
    return math.tanh(max(change, 0.0) / scale)
