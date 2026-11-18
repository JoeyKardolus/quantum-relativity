"""Carry on: a half goes on with the fluid of its cell, and friction takes its direction bit by bit.

Contract: `carry_on(state, rows)` takes this step's crossings (`read_sent`) and returns,
per row of `rows`, how many of the quanta sent were some open split's (int64). It writes
only the store. A split is a claim on quanta of its halves' cells, so when a cell's
fluid crosses on, its halves go with it. For one cell that held `before` quanta and sent
`sent` of them, the quanta it held are laid in a row, each half's first and the quanta
no split claims last, and the `sent` that left are picked evenly along that row:
  go     a half of `quanta` gives `floor((sent * upto + offset) / before)` less the same
         for the halves before it, `upto` the halves' quanta counted so far and `offset`
         one draw below `before` for the cell (`SPLIT_CARRY`). That is `quanta * sent /
         before` to the nearest whole number, never more than the half, and all the
         cell's halves together never more than it sent or than it keeps.
  deal   the quanta that left are a ring, each taking volume's share one stretch of it,
         the ring turned by one draw for the cell (`SPLIT_DEAL`); a half's quanta lie in
         the stretches they fall in. So every quantum that went on is in exactly one
         taking volume, no crossing carries more claimed quanta than it sent, and on
         average each volume gets its share. They are the split's halves there now: a
         split spreads on from cell to cell for as long as it is open.
  cool   of the quanta that went on through a way, the joint's friction took the
         direction of that way's share `lost / sent`. A quantum loses its direction
         once: only the split's share that still had it (`directed` over its quanta)
         counts. `floor(their sum + uniform)` (draw `SPLIT_COOL`, by split and cell)
         comes off the split's `directed`, never below 0. So a split loses its momentum crossing by
         crossing, and settles when none is left (`settle`).
A pointer whose half went on whole is drawn again by share (`stand_pointers`).
So no quantum is ever claimed twice, however long a split stays open. Counted `carried`
(quanta of halves that went on) and `cooled` (direction lost). Does nothing while no
split is open or nothing crossed.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_rows
from physics.engine.axiom_1_energy.definitions.passes import SPLIT_CARRY, SPLIT_COOL, SPLIT_DEAL
from physics.engine.axiom_7_quantum.definitions.split_store import PART_COLUMNS, SPLIT_COLUMNS, amount_of
from physics.engine.axiom_7_quantum.logic.set_splits import set_splits
from physics.engine.axiom_7_quantum.logic.stand_pointers import AFTER_CROSSING, stand_pointers

_SOURCE_BITS = 32


def _pairs(rows: dict, cell: np.ndarray) -> tuple:
    """Every (half, crossing out of its cell) pair: the half's row and the crossing's row."""
    first = np.searchsorted(rows["giver"], cell, side="left")
    count = np.searchsorted(rows["giver"], cell, side="right") - first
    half = np.repeat(np.arange(len(cell)), count)
    within = np.arange(len(half)) - np.repeat(np.cumsum(count) - count, count)
    return half, first[half] + within


def _cell_draws(state, pass_id: int, cells: int) -> np.ndarray:
    """One draw per cell id."""
    return draw_uniform_rows(state.seed, state.step, pass_id, np.arange(cells))


def _stretches(state, store: dict, sent: np.ndarray, before: np.ndarray) -> tuple:
    """Per half: where its going quanta lie (from, to) among the quanta its cell sent, picked evenly."""
    cell, quanta = store["part_cell"], store["part_quanta"]
    order = np.argsort(cell, kind="stable")
    running = np.cumsum(quanta[order])
    starts = np.flatnonzero(np.r_[True, np.diff(cell[order]) != 0])
    base = np.repeat(np.r_[0, running[starts[1:] - 1]], np.diff(np.r_[starts, len(order)]))
    upto = np.empty(len(cell), dtype=np.int64)
    upto[order] = running - base
    offset = (_cell_draws(state, SPLIT_CARRY, len(sent)) * before).astype(np.int64)[cell]
    whole = np.maximum(before[cell], 1)
    return (sent[cell] * (upto - quanta) + offset) // whole, (sent[cell] * upto + offset) // whole


def _overlap(low: np.ndarray, high: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """How many whole places two stretches share."""
    return np.maximum(np.minimum(high, end) - np.maximum(low, start), 0)


def _dealt(state, rows: dict, half: np.ndarray, crossing: np.ndarray, low: np.ndarray, high: np.ndarray,
           sent: np.ndarray) -> np.ndarray:
    """Per pair: how many of the half's going quanta lie in that crossing's stretch of the turned ring."""
    ring = np.maximum(sent[rows["giver"][crossing]], 1)
    running = np.cumsum(rows["sent"])
    first = np.searchsorted(rows["giver"], rows["giver"], side="left")
    before_mine = (running - rows["sent"] - np.where(first > 0, running[np.maximum(first, 1) - 1], 0))[crossing]
    turned = (_cell_draws(state, SPLIT_DEAL, len(sent)) * sent).astype(np.int64)[rows["giver"][crossing]]
    start = (before_mine + turned) % ring
    end = start + rows["sent"][crossing]
    return (_overlap(low[half], high[half], start, np.minimum(end, ring))
            + _overlap(low[half], high[half], 0, np.maximum(end - ring, 0)))


def _draws(state, pass_id: int, store: dict) -> np.ndarray:
    """One draw per half, by its split and its cell."""
    return draw_uniform_rows(state.seed, state.step, pass_id, (store["part_source"] << _SOURCE_BITS) | store["part_cell"])


def _cooled(state, store: dict, rows: dict, half: np.ndarray, crossing: np.ndarray, moved: np.ndarray,
            going: np.ndarray) -> np.ndarray:
    """Per open split: the quanta of its halves whose direction the joints' friction took as they went on."""
    split = np.searchsorted(store["source"], store["part_source"])
    moving = (store["directed"] / np.maximum(amount_of(store), 1))[split]
    cooling = np.zeros(len(going), dtype=np.float64)
    np.add.at(cooling, half, moved * (rows["lost"][crossing] / rows["sent"][crossing]))
    cooling *= moving
    cooled = np.minimum(np.floor(cooling + _draws(state, SPLIT_COOL, store)).astype(np.int64), going)
    lost = np.zeros(len(store["source"]), dtype=np.int64)
    np.add.at(lost, split, cooled)
    return np.minimum(lost, store["directed"])


def _count(store: dict, carried: int, cooled: int) -> None:
    """Add this step's carried and cooled quanta to the store's counts."""
    for name, count in (("carried", carried), ("cooled", cooled)):
        store["step"][name] += count
        store["total"][name] += count


def carry_on(state, rows: dict) -> np.ndarray:
    """Carry every half on with what its cell sent; returns the quanta of open splits in each crossing."""
    store = state.quantum
    claimed = np.zeros(len(rows["key"]), dtype=np.int64)
    if not len(store["part_source"]) or not len(rows["key"]):
        return claimed
    cell, quanta = store["part_cell"], store["part_quanta"]
    sent = np.zeros(int(state.count), dtype=np.int64)
    np.add.at(sent, rows["giver"], rows["sent"])
    low, high = _stretches(state, store, sent, state.host("quanta_before").astype(np.int64)[:len(sent)])
    going = high - low
    half, crossing = _pairs(rows, cell)
    moved = _dealt(state, rows, half, crossing, low, high, sent)
    np.add.at(claimed, crossing, moved)
    lost = _cooled(state, store, rows, half, crossing, moved, going)
    splits = {name: store[name].copy() for name in SPLIT_COLUMNS + ("reacted",)}
    splits["directed"] = splits["directed"] - lost
    parts = dict(zip(PART_COLUMNS, (np.concatenate([store["part_source"], store["part_source"][half]]),
                                    np.concatenate([cell, rows["taker"][crossing]]),
                                    np.concatenate([quanta - going, moved]))))
    set_splits(store, splits, parts)
    stand_pointers(state, AFTER_CROSSING)
    _count(store, int(going.sum()), int(lost.sum()))
    return claimed
