"""Joint push: how hard each joint pushes back on its connections (bible rule 8).

Contract: `push(squeeze, settings)` takes the squeeze on each joint (the mean
log-shortening its connections got from pull this step) and returns
push_j = min(joint_push * squeeze_j, push_limit). A joint asked for more than
push_limit gives way (joint collapse, decided in step.py from the same numbers).
"""
from __future__ import annotations

import numpy as np

from physics.rules.settings import RuleSettings


def push(squeeze: np.ndarray, settings: RuleSettings) -> np.ndarray:
    """Per-joint push: proportional to the squeeze, never more than push_limit."""
    return np.minimum(settings.joint_push * squeeze, settings.push_limit)
