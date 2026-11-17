"""Speed handover: a joint that stops swinging hands its speed to the joints it shared springs with.

Contract: `speed_handover(state, settings, universe, stopped)` takes the cell ids of
joints of one universe that swing no longer (stopped at the collapse limit by the swing,
past it already, or collapsing: joint_swing.py, collapse.py) and shares each one's speed
(state.speed, its size change per unit of counter time) out over the swinging joints it
shares an energy volume with, in proportion to their clock rates (state.rate: the
weights of its springs with them), then sets its own to 0. The springs only ever trade
size between joints (bible rule 5; joint_swing.py: what one joint gains its neighbour
loses), so the swinging joints' speeds add up to nothing; a joint stopped dead would
leave its neighbours' speeds adding up to its own, the other way, and the joints' summed
size drifting for good (before 2026-11-17: by 4,065 boxes in one step of a universe of
2.3 million). A swinging joint is a joint holding quanta whose size is not below its
collapse limit (quanta / collapse_packing) and that is not in `stopped`. A stopped joint
with no swinging neighbour loses its speed. Reads state.kind, between, rate, size,
cell_quanta; writes state.speed. The sums run in cell id order. Returns None.
"""
from __future__ import annotations

import numpy as np

from physics.engine.settings import Settings
from physics.engine.state import State


def speed_handover(state: State, settings: Settings, universe: int, stopped: np.ndarray) -> None:
    """Share each stopped joint's speed over its swinging neighbours by clock rate and set its own to 0."""
    stopped = np.asarray(stopped, dtype=np.int64)
    speed = state.speed[universe]
    stopped = stopped[speed[stopped] != 0.0]
    if stopped.size == 0:
        return
    sides = state.between[universe]
    rate = state.rate[universe]
    swinging = (state.kind[universe] == -1) & (state.cell_quanta[universe] > 0)
    swinging &= state.size[universe] >= state.cell_quanta[universe] / settings.collapse_packing
    swinging[stopped] = False
    takers = np.flatnonzero(swinging)
    capacity = rate.size
    rates = np.zeros(capacity)
    given = np.zeros(capacity)
    for pick in range(2):
        volume = sides[takers, pick]
        rates += np.bincount(volume[volume >= 0], weights=rate[takers][volume >= 0], minlength=capacity)
    for pick in range(2):
        volume = sides[stopped, pick]
        there = volume >= 0
        total = sum(np.where(sides[stopped, side] >= 0, rates[np.maximum(sides[stopped, side], 0)], 0.0) for side in range(2))
        share = np.where(total > 0.0, speed[stopped] / np.where(total > 0.0, total, 1.0), 0.0)
        given += np.bincount(volume[there], weights=share[there], minlength=capacity)
    gain = np.zeros(takers.size)
    for pick in range(2):
        volume = sides[takers, pick]
        gain += np.where(volume >= 0, given[np.maximum(volume, 0)], 0.0)
    state.speed = state.speed.copy()
    state.speed[universe, takers] += rate[takers] * gain
    state.speed[universe, stopped] = 0.0
