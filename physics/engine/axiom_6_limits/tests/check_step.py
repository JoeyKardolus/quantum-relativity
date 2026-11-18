"""Check step: axiom 6 counted on the state as it stands after one step.

Contract: `check_step(state, settings) -> dict`; it writes nothing. What each joint held
at the last call is remembered here, per state, and forgotten with the state. At the
first call, after step 1, what each joint held before the step is what it was made with
(`fresh_quanta`: a start's joint holds its fresh quanta until something is made), so the
first step is compared like every other.
  made                minus quanta squeezed joints made this step
  making_unsqueezed   joints that made something at a squeeze of 1 or less; must be 0
  joints_below_fresh  living joints holding less than they were made with; must be 0
  joints_fell         joints, living then and now, that hold less than at the last call;
                      must be 0 (a joint's energy never falls)
  joints_rose_unmade  joints, living then and now, whose quanta rose by anything but what
                      they made this step; must be 0 (it rises only while squeezed, made
                      out of the joint itself)
  joints_past_limit   living joints holding `collapse_thin` times their fresh quanta or
                      more after the step's events; must be 0 (squeezed too far it
                      collapses)
  joint_steps_compared   how many joints, living then and now, the two counts above
                      were read on this step (0 would mean they compared nothing)
  collapses, merges, tears, short_takes, short_quanta   the counts kept since the start
  over_tear_length    energy volumes longer than `tear_length` now
  events              this step's event records (copies)
"""
from __future__ import annotations

import copy
import weakref

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT

_before: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _since_last(state, kind: np.ndarray, quanta: np.ndarray, made: np.ndarray) -> tuple:
    """Joints living at the last call and now: how many hold less, how many rose by other than `made`, how many there are."""
    last = _before.get(state)
    _before[state] = (kind.copy(), quanta.copy())
    if last is None and state.step == 1:
        last = (state.host("made_kind"), state.host("fresh_quanta"))
    if last is None:
        return 0, 0, 0
    known = min(last[0].size, kind.size)
    both = (last[0][:known] == JOINT) & (kind[:known] == JOINT)
    change = quanta[:known] - last[1][:known]
    return int((both & (change < 0)).sum()), int((both & (change != made[:known])).sum()), int(both.sum())


def check_step(state, settings) -> dict:
    """Axiom 6's counts for the step that just ran."""
    kind, made, squeeze = state.host("kind"), state.host("made"), state.host("squeeze")
    quanta, fresh, length = state.host("cell_quanta"), state.host("fresh_quanta"), state.host("length")
    joints, volumes = kind == JOINT, kind == ENERGY_VOLUME
    fell, rose_unmade, compared = _since_last(state, kind, quanta, made)
    return {
        "made": int(made.sum()),
        "making_unsqueezed": int(np.count_nonzero((made > 0) & ~(squeeze > 1.0))),
        "joints_below_fresh": int(np.count_nonzero(joints & (quanta < fresh))),
        "joints_fell": fell,
        "joints_rose_unmade": rose_unmade,
        "joint_steps_compared": compared,
        "joints_past_limit": int(np.count_nonzero(joints & (fresh > 0) & (quanta >= settings.collapse_thin * fresh))),
        "collapses": int(state.collapses),
        "tears": int(state.tears),
        "merges": int(state.merges),
        "short_takes": int(state.short_takes),
        "short_quanta": int(state.short_quanta),
        "over_tear_length": int(np.count_nonzero(volumes & (length > settings.tear_length))),
        "events": copy.deepcopy(list(state.changed)),
    }
