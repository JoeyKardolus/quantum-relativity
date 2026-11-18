"""Phase arrays: the state axiom 7 owns, each box's phase and the open splits.

Contract: `ARRAYS = {name: (dtype, shape, each_step)}` and `HOST = {name: default}`.
  phase    float64 per box, radians in (-pi, pi]: the box's fluid is the wave
           `sqrt(abs(quanta)) e^{i phase}`. Because every box carries a phase, waves add
           and cancel wherever fluid meets fluid. Wrapped at every write.
  quantum  host: None until `start_store` puts the store of open splits there
           (split_store.py)
"""
from __future__ import annotations

ARRAYS = {
    "phase": ("float64", "box", False),
}

HOST = {
    "quantum": None,
}
