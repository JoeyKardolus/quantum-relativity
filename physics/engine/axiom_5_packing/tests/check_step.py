"""Check step: axiom 5 read off one state, as plain numbers for the per-step records.

Contract: `check_step(state, settings) -> dict`, host only, changes nothing. Called after
a whole step (so `press_joints` has run in it). The contract's keys:
  slowest_rate      the lowest clock rate of any living cell
  deepest_pull      the lowest pull of any living cell
  pull_mean         the mean pull of the living cells (the level all cells share; it
                    sinks faster every step (as the step squared) and no rule reads it)
  squeezed_joints   {"face", "edge", "corner"}: living joints whose squeeze is above 1
  largest_squeeze   {"face", "edge", "corner"}: the largest squeeze of each type
A joint's type is read off its hardest way's mouth, as the collapse reads it: the number
of indices that mouth steps along (1 face, 2 edge, 3 corner). A joint with no way has no
type and is in none of the counts.
Added here, for the clause tests:
  stale_cells           living cells whose length, packing or clock rate is not, to five
                        places, the law of the quanta they hold now, worked again
                        here on the host (must be 0: a size or a clock left over from
                        before an event or a crossing would show here)
  slow_plus_share       the share of the plus energy in volumes whose clock runs below
                        a thousandth of full rate: how much energy has all but stopped
  pull_follows_packing  over the living energy volumes, how the pull goes with packing,
                        from -1 to 1: below 0 means the more packed sit deeper (5c);
                        0 while they are all alike
  largest_pull_change   the largest change of any cell's pull in the last step, the
                        level all cells share taken out
  gain_goes_with_depth  over the living energy volumes, how what each gained by this
                        step's crossing goes with how deep it sits in the pull, from -1
                        to 1: above 0 means energy moved toward the deeper pull
  fuller_gain           the plus quanta the fuller half of the volumes (by what they
                        held before the crossing) gained from the emptier half this step
  largest_turn          the largest turn, in radians, the pull gave any box this step
  pull_not_finite       living cells whose pull is not a number (must be 0)
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.direction_codes import step_of_host
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT, NO_CELL

JOINT_TYPES = ("face", "edge", "corner")
_TYPE_OF_CODE = np.array([sum(part != 0 for part in step_of_host(code)) for code in range(27)])


def _joint_types(state, joints: np.ndarray) -> np.ndarray:
    """For each of the joints: 1 face, 2 edge, 3 corner by its hardest way's mouth; 0 with no way."""
    hardest = state.host("squeezed_way")[joints]
    types = np.zeros(len(joints), dtype=np.int64)
    has_way = hardest >= 0
    types[has_way] = _TYPE_OF_CODE[state.host("way_mouth_a")[hardest[has_way]]]
    return types


def _stale(state, settings, kind: np.ndarray) -> int:
    """Living cells whose length, packing or clock rate is not the law of the quanta they hold now."""
    quanta = state.host("cell_quanta").astype(np.float64)
    held = np.maximum(quanta, 1.0)
    length = np.where(quanta > 0, settings.size_constant / held, np.inf)
    packing = np.where(quanta > 0, quanta / length ** 3, 0.0)
    rate = 1.0 / (1.0 + packing / settings.clock_packing)
    off = ~np.isclose(state.host("packing"), packing, rtol=1e-5, atol=0.0) \
        | ~np.isclose(state.host("rate"), rate, rtol=1e-5, atol=1e-30) \
        | ~np.isclose(state.host("length").astype(np.float64), length, rtol=1e-5, atol=0.0)
    return int((off & (kind != NO_CELL)).sum())


def _goes_with(first: np.ndarray, second: np.ndarray) -> float:
    """How two rows of numbers go together, -1 to 1; 0 when either does not vary."""
    if len(first) < 2 or first.std() == 0.0 or second.std() == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _largest_change(change: np.ndarray, cells: np.ndarray) -> float:
    """The largest change of the pull among some cells once the change they all share is taken out."""
    if not cells.any():
        return 0.0
    return float(np.abs(change[cells] - change[cells].mean()).max())


def _fuller_gain(before: np.ndarray, gained: np.ndarray) -> int:
    """What the fuller half of the volumes gained this step."""
    if before.size < 2:
        return 0
    return int(gained[before > np.median(before)].sum())


def _slow_share(quanta: np.ndarray, rate: np.ndarray, volumes: np.ndarray) -> float:
    """The share of the plus energy in volumes whose clock runs below a thousandth of full rate."""
    plus = float(quanta[volumes].sum())
    return float(quanta[volumes & (rate < 1e-3)].sum()) / plus if plus else 0.0


def check_step(state, settings) -> dict:
    """Axiom 5's numbers for one step's record."""
    kind = state.host("kind")
    living = kind != NO_CELL
    rate, pull = state.host("rate"), state.host("pull").astype(np.float64)
    quanta, length, packing = state.host("cell_quanta"), state.host("length"), state.host("packing")
    squeeze = state.host("squeeze")
    joints = np.flatnonzero(kind == JOINT)
    types = _joint_types(state, joints)
    volumes = kind == ENERGY_VOLUME
    finite = np.isfinite(pull)
    before = state.host("quanta_before").astype(np.float64)
    gained = np.where(volumes, quanta - before, 0.0)
    return {
        "slowest_rate": float(rate[living].min()),
        "deepest_pull": float(pull[living & finite].min(initial=0.0)),
        "pull_mean": float(pull[living & finite].mean()) if (living & finite).any() else 0.0,
        "squeezed_joints": {name: int((squeeze[joints[types == number]] > 1.0).sum())
                            for number, name in enumerate(JOINT_TYPES, start=1)},
        "largest_squeeze": {name: float(squeeze[joints[types == number]].max(initial=0.0))
                            for number, name in enumerate(JOINT_TYPES, start=1)},
        "stale_cells": _stale(state, settings, kind),
        "slow_plus_share": _slow_share(quanta, rate, volumes),
        "pull_follows_packing": _goes_with(packing[volumes & finite], pull[volumes & finite]),
        "largest_pull_change": _largest_change(pull - state.host("pull_old"), living & finite),
        "gain_goes_with_depth": _goes_with(gained[volumes & finite], -pull[volumes & finite]),
        "fuller_gain": _fuller_gain(before[volumes], gained[volumes]),
        "largest_turn": float(state.host("pull_felt").max(initial=0.0)),
        "pull_not_finite": int((living & ~finite).sum()),
    }
