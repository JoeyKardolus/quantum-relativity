"""Score samples: the fit's score of one universe's run, 0 when nothing happens and at most 1.

Contract: `score_samples(samples, held_share=0.0)` returns a dict. `samples` are
`measure_cells` dicts in time order, each with "tau", the observer's time at the sample,
and "gap", the distance between the end's lumps then (None when the run ends with fewer
than two); the first sample is the start. `held_share` is the run's held share.

  score = hold * (E + S) / 2          every factor between 0 and 1

  E     expansion shape: the mean of `length_shape` for two lengths, the size of the
        universe and the gap between its lumps (0 when there are no two lumps)
  S     structure: the mean of the four `structure_parts`
  hold  `hold_factor` of the last sample
Every part is a change since the start, so a universe where nothing happens scores
exactly 0; tanh caps every part, so no runaway earns more than steady growth does.
Returns "score", "E", "S", "hold", "E_size", "E_gap", "growth", "dip", "gap_growth",
"gap_dip" and the four structure parts.
Raises ValueError for fewer than MIN_SAMPLES samples or an observer's time that does not
advance.
"""
from __future__ import annotations

import numpy as np

from physics.fit.score.definitions.hold_factor import hold_factor
from physics.fit.score.definitions.length_shape import length_shape
from physics.fit.score.definitions.score_marks import MIN_SAMPLES
from physics.fit.score.definitions.structure_parts import structure_parts

_NO_GAP = {"shape": 0.0, "growth": 0.0, "dip": 0.0}


def _observer_times(samples: list) -> np.ndarray:
    """The samples' observer times; raises when they are too few or do not advance."""
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"need at least {MIN_SAMPLES} samples, got {len(samples)}")
    tau = np.array([sample["tau"] for sample in samples], dtype=np.float64)
    if not np.all(np.diff(tau) > 0.0):
        raise ValueError("the observer's time must advance between samples")
    return tau


def score_samples(samples: list, held_share: float = 0.0) -> dict:
    """Score, its expansion and structure halves, the hold factor and every part, from a run's samples."""
    tau = _observer_times(samples)
    size = length_shape(tau, [sample["size"] for sample in samples])
    gaps = [sample.get("gap") for sample in samples]
    gap = length_shape(tau, gaps) if all(value is not None for value in gaps) else _NO_GAP
    parts = structure_parts(samples[0], samples[-1], held_share)
    expansion, structure = 0.5 * (size["shape"] + gap["shape"]), sum(parts.values()) / len(parts)
    hold = hold_factor(samples[-1]["alive"], samples[-1]["spent_share"])
    return {"score": hold * 0.5 * (expansion + structure), "E": expansion, "S": structure, "hold": hold,
            "E_size": size["shape"], "E_gap": gap["shape"], "growth": size["growth"], "dip": size["dip"],
            "gap_growth": gap["growth"], "gap_dip": gap["dip"], **parts}
