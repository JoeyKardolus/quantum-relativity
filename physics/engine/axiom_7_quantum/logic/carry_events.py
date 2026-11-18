"""Carry events: this step's collapses and tears change the cells some halves lie in.

Contract: `carry_events(state)` reads `state.changed` (in order) and writes only the
store. Does nothing while no split is open or nothing changed.
  collapse with two volumes (`kept`, `gone`): every half in `gone` now lies in `kept`,
    a pointer on `gone` stands on `kept`, and a split whose source was `gone` is `kept`'s.
  collapse with one volume left (`kept == gone`): nothing moves.
  tear (`volume`, `joint`, `new_volume`, `quanta` of the three pieces): every half in the
    torn volume is shared over the three pieces in proportion to their quanta
    (`whole_shares`); the joint's share turned minus and leaves the split
    (`torn_quanta`); a pointer that stood on the torn volume lands on the low or the new
    piece by their shares (draw `SPLIT_POINTER`, number 2 ** 31 + the event's place).
  Each is something reacting with a half: every split with a half in a cell named, or
  whose source is named, is marked `reacted`.
Afterwards the tables are merged again (`set_splits`; a split left with nothing is
counted `ended`) and a pointer whose cell holds no half of its split any more is drawn
again by share (`stand_pointers`).
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_host
from physics.engine.axiom_1_energy.definitions.passes import SPLIT_POINTER
from physics.engine.axiom_1_energy.definitions.whole_shares import whole_shares
from physics.engine.axiom_7_quantum.definitions.split_store import PART_COLUMNS, SPLIT_COLUMNS
from physics.engine.axiom_7_quantum.logic.set_splits import set_splits
from physics.engine.axiom_7_quantum.logic.stand_pointers import AFTER_EVENTS, stand_pointers

_SOURCE_BITS = 32
_TEAR_DRAWS = 2 ** 31


def _react(splits: dict, parts: dict, cells: tuple) -> None:
    """Mark every split that has a half in one of `cells` or whose source is one of them."""
    touched = parts["part_source"][np.isin(parts["part_cell"], cells)]
    splits["reacted"] |= np.isin(splits["source"], touched) | np.isin(splits["source"], cells)


def _after_collapse(splits: dict, parts: dict, event: dict) -> None:
    """Move everything that named the vanished volume onto the kept one."""
    kept, gone = event.get("kept"), event.get("gone")
    if kept is None:
        return
    _react(splits, parts, (kept, gone))
    if kept == gone:
        return
    for column in (parts["part_cell"], parts["part_source"], splits["pointer"], splits["source"]):
        column[column == gone] = kept


def _after_tear(state, splits: dict, parts: dict, event: dict, place: int) -> int:
    """Share every half in a torn volume over its pieces; returns the quanta that turned minus."""
    volume, new_volume = int(event["volume"]), int(event["new_volume"])
    _react(splits, parts, (volume,))
    pieces = [int(quanta) for quanta in event["quanta"]]
    torn = 0
    for row in np.flatnonzero(parts["part_cell"] == volume):
        source = int(parts["part_source"][row])
        low, flipped, high = whole_shares(int(parts["part_quanta"][row]), pieces)
        parts["part_quanta"][row] = low
        for name, value in zip(PART_COLUMNS, (source, new_volume, high)):
            parts[name] = np.append(parts[name], value)
        torn += flipped
        on_torn = (splits["source"] == source) & (splits["pointer"] == volume)
        uniform = draw_uniform_host(state.seed, state.step, SPLIT_POINTER, (source << _SOURCE_BITS) | (_TEAR_DRAWS + place))
        if low + high > 0 and uniform * (low + high) >= low:
            splits["pointer"][on_torn] = new_volume
    return torn


def carry_events(state) -> None:
    """Carry this step's collapses and tears into the open splits' tables."""
    store = state.quantum
    if not len(store["source"]) or not state.changed:
        return
    splits = {name: store[name].copy() for name in SPLIT_COLUMNS + ("reacted",)}
    parts = {name: store[name].copy() for name in PART_COLUMNS}
    torn = 0
    for place, event in enumerate(state.changed):
        if event["what"] == "collapse":
            _after_collapse(splits, parts, event)
        elif event["what"] == "tear":
            torn += _after_tear(state, splits, parts, event, place)
    ended = set_splits(store, splits, parts)
    stand_pointers(state, AFTER_EVENTS)
    for name, count in (("torn_quanta", torn), ("ended", ended)):
        store["step"][name] += count
        store["total"][name] += count
