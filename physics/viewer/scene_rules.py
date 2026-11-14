"""The rules every viewer scene runs under: all rules on, drift balancing the even background.

Contract: `scene_rules(background, seed)` returns `all_rules_on` with attraction
set so the median pull of `background` (the scene's even ball, before any lump)
shortens a connection by ATTRACTION per step, and drift set by
`balancing_drift` on that background (rng seeded from `seed`): the one
calibration, which keeps even space the same size and, like Einstein's static
universe, need not stay balanced once lumps add pull.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.network import Network
from physics.rules import RuleSettings, all_rules_on, balancing_drift, pull

ATTRACTION = 0.01       # log-shortening per step at the background's median pull


def scene_rules(background: Network, seed: int) -> RuleSettings:
    """All rules on, attraction scaled to the background pull, balancing drift."""
    median_pull = float(np.median(pull(background, background.energy, background.length)))
    settings = all_rules_on(attraction=ATTRACTION / median_pull)
    return replace(settings, drift=balancing_drift(background, settings, np.random.default_rng(seed)))
