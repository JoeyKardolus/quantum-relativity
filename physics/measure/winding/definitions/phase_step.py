"""Phase step: how far the phase turns going from one box to the next.

Contract: `phase_step(first, second)` takes two complex numpy arrays of one shape (a
box's number is its amount times exp(i phase)) and returns, as float64, the phase of
`second` minus the phase of `first`, brought into (-pi, pi]: the short way round.
Because every step is the short way, the steps round a closed loop of boxes add up to
a whole number of full turns exactly. Where either number is 0 the result is 0 and
means nothing: a box holding nothing has no phase, and the caller must leave it out.
"""
from __future__ import annotations

import numpy as np


def phase_step(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The phase change from `first` to `second`, the short way round."""
    return np.angle(np.conj(first) * second)
