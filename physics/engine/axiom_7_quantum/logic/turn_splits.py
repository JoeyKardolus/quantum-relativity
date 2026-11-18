"""Turn splits: every split's own oscillation goes on, and at each whole turn its pointer jumps.

Contract: `turn_splits(state, settings)` writes only `pointer` of the store. A split
born in this step has not turned yet. For an older one:
  speed   drawn at the split: `speed_draw` of the draw `SPLIT_SPEED` of the step it was
          born in and its volume (number 0), between `settings.slowest_speed` and
          `settings.fastest_speed`, in turns per step of the engine. It is worked again
          here every step and kept nowhere.
  turns   `speed * age`, `age` the steps since the split: counted against no cell's
          clock. The whole turns passed in this step are
          `floor(speed * age) - floor(speed * (age - 1))`.
  jump    at each whole turn the pointer jumps to one of the halves' cells with that
          cell's share of the energy as its chance (`pointer_law`). Each jump is a fresh
          draw by share, so only the last of a step stands: one draw is made, `SPLIT_POINTER`
          numbered by the count of whole turns the split has made (1 or more).
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_rows
from physics.engine.axiom_1_energy.definitions.passes import SPLIT_POINTER, SPLIT_SPEED
from physics.engine.axiom_7_quantum.definitions.pointer_law import pointer_law
from physics.engine.axiom_7_quantum.definitions.speed_draw import speed_draw

_SOURCE_BITS = 32
_TURN_NUMBERS = 2 ** 31 - 1


def _turns(state, settings, store: dict) -> tuple:
    """Per split: the whole turns made before this step and by the end of it."""
    before = np.zeros(len(store["source"]), dtype=np.int64)
    after = np.zeros(len(store["source"]), dtype=np.int64)
    for born in np.unique(store["born"]):
        age = int(state.step) - int(born)
        if age < 1:
            continue
        rows = store["born"] == born
        uniform = draw_uniform_rows(state.seed, int(born), SPLIT_SPEED, store["source"][rows] << _SOURCE_BITS)
        speed = speed_draw(uniform, settings.slowest_speed, settings.fastest_speed)
        before[rows] = np.floor(speed * (age - 1)).astype(np.int64)
        after[rows] = np.floor(speed * age).astype(np.int64)
    return before, after


def turn_splits(state, settings) -> None:
    """Turn every older split by one step and let its pointer jump where a whole turn was passed."""
    store = state.quantum
    if not len(store["source"]):
        return
    before, after = _turns(state, settings, store)
    jumped = after > before
    sources = store["source"][jumped]
    number = 1 + (after[jumped] - 1) % _TURN_NUMBERS
    uniform = draw_uniform_rows(state.seed, state.step, SPLIT_POINTER, (sources << _SOURCE_BITS) | number)
    store["pointer"][jumped] = pointer_law(store["part_source"], store["part_cell"], store["part_quanta"], sources,
                                           uniform)
