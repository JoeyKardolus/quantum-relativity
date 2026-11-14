"""One rules step that reports a breakdown instead of passing on an unreadable network.

Contract: `safe_step(network, settings)` returns (stepped, None) when the step
gives a readable network, else (None, reason): the step raised ValueError,
or a length came out non-finite or not above zero (Dijkstra cannot read it).
"""
from __future__ import annotations

import numpy as np

from physics.network import Network
from physics.rules import RuleSettings, step


def safe_step(network: Network, settings: RuleSettings) -> tuple[Network | None, str | None]:
    """The stepped network, or None and why it could not be read."""
    try:
        with np.errstate(all="ignore"):
            stepped = step(network, settings)
    except ValueError as error:
        return None, str(error)
    if not (np.all(np.isfinite(stepped.length)) and np.all(stepped.length > 0)):
        return None, "a connection length reached zero or infinity"
    return stepped, None
