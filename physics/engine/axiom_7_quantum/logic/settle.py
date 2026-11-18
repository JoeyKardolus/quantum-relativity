"""Settle: a split that was reacted with, or has no momentum left, settles whole in its pointer's cell.

Contract: `settle(state)` reads the store, `kind`, `cell_quanta` (this step's),
and `heat_quanta`, and writes the store and, through `draw_whole`, `quanta` and
`heat_quanta`.
  Which settle: every split marked `reacted`, every split with a half in a cell of
  `detected` (a caller's detector; the set is then emptied), cause reaction; and every
  other split whose `directed` is 0, cause momentum.
  How: every half that is not in the pointer's cell is drawn into it (`draw_whole`, the
  direct pull between the halves: at once, at any distance). All settlings of a step
  are drawn together. Settling is coming to rest: what is drawn joins the fluid of the
  pointer's cell box by box at the phase that fluid has, so the cell's own swirl stands.
  It comes with its share of the giving cell's heat (of all a cell gives, `given * heat // holds` is heat,
  shared over its moves in whole numbers).
  Not there: when the halves to be drawn out of one cell are more than it holds (it sent
  them on, or a collapse took them), they are cut to what it holds by `whole_shares`,
  and a half in a cell that is no living energy volume is not drawn; the cut is booked
  as `short_quanta`.
The settled splits leave the store: nothing of them is left to turn or to be reacted
with. `store["settled"]` gets one row each (`source`, `cell`, `amount` drawn together
there with the half that already lay there, `short`, `cause`) added to it, and
`store["step"]` and `store["total"]` are raised by what settled.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.whole_shares import whole_shares
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME
from physics.engine.axiom_7_quantum.definitions.split_store import (MOMENTUM, PART_COLUMNS, REACTION,
                                                                     SETTLED_COLUMNS, SPLIT_COLUMNS)
from physics.engine.axiom_7_quantum.logic.draw_whole import draw_whole


def _settling(store: dict) -> np.ndarray:
    """Which open splits settle now: reacted with, detected, or out of momentum."""
    if store["detected"]:
        seen = store["part_source"][np.isin(store["part_cell"], sorted(store["detected"]))]
        store["reacted"] |= np.isin(store["source"], seen)
    store["detected"] = set()
    return store["reacted"] | (store["directed"] == 0)


def _drawn(state, cell: np.ndarray, quanta: np.ndarray, moving: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Per half: the whole quanta that can be drawn, cut where a cell holds less than is asked of it."""
    drawn = quanta.copy()
    drawn[moving & (state.host("kind")[cell] != ENERGY_VOLUME)] = 0
    asked = np.zeros(len(held), dtype=np.int64)
    np.add.at(asked, cell[moving], drawn[moving])
    for short_cell in np.flatnonzero(asked > held):
        rows = np.flatnonzero(moving & (cell == short_cell))
        drawn[rows] = whole_shares(int(held[short_cell]), [int(value) for value in drawn[rows]])
    return drawn


def _heat_drawn(cell: np.ndarray, drawn: np.ndarray, heat: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Per move: how many of its quanta are heat. A cell gives `all it gives * heat // holds`, shared over its moves."""
    order = np.argsort(cell, kind="stable")
    cells, first, row_of = np.unique(cell[order], return_index=True, return_inverse=True)
    gives = np.zeros(len(cells), dtype=np.int64)
    np.add.at(gives, row_of, drawn[order])
    leaves = gives * heat[cells] // np.maximum(held[cells], 1)
    share = drawn[order] * leaves[row_of] // np.maximum(gives[row_of], 1)
    owed = leaves.copy()
    np.subtract.at(owed, row_of, share)
    share += (np.arange(len(order)) - first[row_of]) < owed[row_of]
    moved = np.zeros(len(cell), dtype=np.int64)
    moved[order] = share
    return moved


def _plan(state, cell: np.ndarray, target: np.ndarray, drawn: np.ndarray, held: np.ndarray) -> dict:
    """The three per-cell numbers `draw_whole` takes, for moves of `drawn` quanta from `cell` to `target`."""
    cells = len(held)
    heat = _heat_drawn(cell, drawn, state.host("heat_quanta").astype(np.int64), held)
    plan = {name: np.zeros(cells, dtype=np.int64) for name in ("take", "give", "heat_change")}
    np.add.at(plan["take"], cell, drawn)
    np.add.at(plan["give"], target, drawn)
    np.add.at(plan["heat_change"], target, heat)
    np.subtract.at(plan["heat_change"], cell, heat)
    return plan


def _book(store: dict, settling: np.ndarray, row_split: np.ndarray, quanta: np.ndarray, drawn: np.ndarray,
          moving: np.ndarray) -> None:
    """Write this step's settlings and counts, and take the settled splits out of the tables."""
    place = np.cumsum(settling) - 1
    amount = np.zeros(int(settling.sum()), dtype=np.int64)
    short = np.zeros(len(amount), dtype=np.int64)
    np.add.at(amount, place[row_split], drawn)
    np.add.at(short, place[row_split], quanta - drawn)
    cause = np.where(store["reacted"][settling], REACTION, MOMENTUM).astype(np.int64)
    rows = (store["source"][settling], store["pointer"][settling], amount, short, cause)
    store["settled"] = {name: np.concatenate([store["settled"][name], row])
                        for name, row in zip(SETTLED_COLUMNS, rows)}
    counts = {"settled_reaction": int(np.count_nonzero(cause == REACTION)),
              "settled_momentum": int(np.count_nonzero(cause == MOMENTUM)), "settled_quanta": int(amount.sum()),
              "drawn_quanta": int(drawn[moving].sum()), "short_quanta": int(short.sum())}
    for name, count in counts.items():
        store["step"][name] += count
        store["total"][name] += count
    kept_parts = ~settling[np.searchsorted(store["source"], store["part_source"])]
    for name in PART_COLUMNS:
        store[name] = store[name][kept_parts]
    for name in SPLIT_COLUMNS + ("reacted",):
        store[name] = store[name][~settling]


def settle(state) -> None:
    """Draw every settling split whole into its pointer's cell and take it out of the store."""
    store = state.quantum
    settling = _settling(store)
    split_of = np.searchsorted(store["source"], store["part_source"])
    rows = settling[split_of] if len(split_of) else np.zeros(0, dtype=bool)
    row_split, cell, quanta = split_of[rows], store["part_cell"][rows], store["part_quanta"][rows]
    target = store["pointer"][row_split]
    moving = cell != target
    held = state.host("cell_quanta").astype(np.int64)
    drawn = _drawn(state, cell, quanta, moving, held)
    if moving.any():
        draw_whole(state, **_plan(state, cell[moving], target[moving], drawn[moving], held))
    _book(store, settling, row_split, quanta, drawn, moving)
