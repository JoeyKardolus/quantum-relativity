"""Switch levels: the once-a-step pass that moves every block to the detail its quietness allows.

Contract: `switch_levels(store)` measures every non-EMPTY block (`measure_quietness`) and
applies the rules, in this order, then syncs the block table:
  1. a block with no space left (all quanta 0) becomes EMPTY and frees its storage;
  2. a busy block (a whirl, or flow above flow_tol) restarts its quiet run and, when
     COARSE or REUSED, comes back to FULL at once: whirls and flow are never coded;
  3. a quiet block adds one to its quiet run; a FULL block whose run reached quiet_steps
     is a candidate: when its exact bytes equal an existing shared copy, or another
     candidate's, it becomes REUSED (that copy); otherwise it is coded COARSE and kept
     COARSE only when every group meets max_rel_error and max_phase_error, else it stays
     FULL (and is tried again next pass).
No rule changes a box's sign or the total whole quanta, plus or minus: FULL and REUSED
are exact, COARSE keeps each group's plus and minus sums exactly (`_kernels.py`).
Returns counts: blocks at each level, and emptied, promoted, shared, coarsened this pass.
"""
from __future__ import annotations

import numpy as np

from physics.ladder._moves import _full_rows, _release, _to_coarse, _to_full, _to_shared
from physics.ladder.level import Level
from physics.ladder.measure_quietness import measure_quietness
from physics.ladder.store import LadderStore


def switch_levels(store: LadderStore) -> dict:
    """Measure every block and move each to the level the rules give; return the counts."""
    coords = np.argwhere(store.level != Level.EMPTY)
    report = {"emptied": 0, "promoted": 0, "shared": 0, "coarsened": 0}
    if coords.shape[0]:
        quiet = measure_quietness(store, coords)
        where = tuple(coords.T)
        empty = quiet["filled"] == 0
        _release(store, coords[empty])
        report["emptied"] = int(empty.sum())
        busy = quiet["busy"] & ~empty
        store.quiet_run[where] = np.where(busy, 0, store.quiet_run[where] + 1)
        lower = busy & (store.level[where] != Level.FULL)
        _to_full(store, coords[lower])
        report["promoted"] = int(lower.sum())
        ready = ~busy & ~empty & (store.level[where] == Level.FULL)
        ready &= store.quiet_run[where] >= store.settings.quiet_steps
        report.update(_demote(store, coords[ready]))
    store.sync()
    for level in Level:
        report[level.name.lower()] = int((store.level == level).sum())
    return report


def _demote(store: LadderStore, coords: np.ndarray) -> dict:
    """Quiet FULL blocks to REUSED when their exact content repeats, else to COARSE when within bound."""
    if coords.shape[0] == 0:
        return {"shared": 0, "coarsened": 0}
    quanta, phase = _full_rows(store, store.slot[tuple(coords.T)])
    keys = [quanta[number].tobytes() + phase[number].tobytes() for number in range(coords.shape[0])]
    seen: dict = {}
    for key in keys:
        seen[key] = seen.get(key, 0) + 1
    repeat = np.array([key in store.shared_key or seen[key] > 1 for key in keys], dtype=bool)
    _to_shared(store, coords[repeat], [key for key, flag in zip(keys, repeat) if flag])
    kept = _to_coarse(store, coords[~repeat])
    return {"shared": int(repeat.sum()), "coarsened": int(kept.sum())}
