"""Limits arrays: the state axiom 6 owns, what each joint was made with and what broke this step.

Contract: `ARRAYS = {name: (dtype, shape, each_step)}` and `HOST = {name: default}`.
Per cell id:
  fresh_quanta  int64, per joint: its quanta when it was made, at the start or by a
                tear; never changed. A joint never holds less than this.
  made          int64, cleared each step, per joint: minus quanta it made this step
  flags         int32, cleared each step: 1 a joint to collapse, 2 a volume to tear
Short:
  found         int32, 2 values, cleared each step: [0] joints flagged to collapse,
                [1] volumes flagged to tear
Host counts since the start: `collapses`, `merges`, `tears`.
"""
from __future__ import annotations

COLLAPSE_FLAG = 1
TEAR_FLAG = 2

ARRAYS = {
    "fresh_quanta": ("int64", "cell", False),
    "made": ("int64", "cell", True),
    "flags": ("int32", "cell", True),
    "found": ("int32", 2, True),
}

HOST = {
    "collapses": 0,
    "merges": 0,
    "tears": 0,
}
