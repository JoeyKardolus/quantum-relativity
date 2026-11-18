"""Check step: axiom 1 checked as a whole on the state as it stands after a step.

Contract: `check_step(state, settings) -> dict` of plain numbers; it writes nothing.
  book_amount, book_plus, book_minus, book_taken   what the totals are less what
                `book_law` says they must be from `state.start_books`; all 0 while every
                change of the total is booked and every collapse took what its joint held
  wrong_boxes   boxes of the wrong sign in a living cell, or not 0 in an id that is no cell
  wrong_cells   ids whose `cell_quanta` is not the sum of their boxes
  empty_cells   living cells holding 0 (zero is empty space: counted, not an error)
How much of the fluid moved is judged on each cell's own clock (axiom 1: "at each moment
of local time"). A cell's `moved` is the quanta that changed box inside it plus the
quanta that left or entered it through its ways, this step; its local time this step is
`step_time * rate`.
  still_cells   living cells in which fewer than one quantum in a hundred moved per unit
                of the cell's own time. A cell so slow or so empty that this bar is under
                one whole quantum a step cannot be judged in one step and is not counted
                here (see `unjudged_share`). A cell made by a tear in this step has not
                lived yet and is not counted
  still_share   the share of all quanta (plus and minus) those cells hold; must be 0
  unjudged_share  the share of all quanta in cells that could not be judged
  moved_share   all quanta that moved this step, inside cells and between them, over all
                quanta: the plain number, on the engine's own step
  acting_ways   living ways between two volumes that hold energy, through a joint that
                holds energy, whose local time this step (the slowest of its three
                clocks) is long enough to judge: a hundredth of the emptier volume's
                quanta per unit of that time is at least one whole quantum a step
  crossed_share   the plus quanta that crossed a joint this step over all plus quanta
  moved_inside, cut_moves              this step: quanta moved inside cells, moves cut at 0
  plus, minus, amount                  the totals themselves
  made, made_total, taken_joint, taken_volumes, short_takes, short_quanta, torn
                                       what changed the total
Raises ValueError when `state.start_books` was never read (`new_universe` reads it).
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.book_law import book_law
from physics.engine.axiom_1_energy.logic.read_books import read_books

_COPIED = ("wrong_boxes", "wrong_cells", "empty_cells", "moved_inside", "cut_moves", "plus", "minus", "amount",
           "made", "made_total", "taken_joint", "taken_volumes", "short_takes", "short_quanta", "torn")
_STILL_BAR = 0.01


def _crossed_by_cell(state, cells: int) -> np.ndarray:
    """Per cell id: the quanta that left or entered it through its ways this step."""
    living = state.host("way_joint") >= 0
    through = (state.host("way_out_a").astype(np.int64) + state.host("way_out_b"))[living]
    crossed = np.zeros(cells, dtype=np.int64)
    np.add.at(crossed, state.host("way_a")[living], through)
    np.add.at(crossed, state.host("way_b")[living], through)
    return crossed


def _acting_ways(state, quanta: np.ndarray, step_time: float) -> int:
    """How many living ways between cells that hold energy have a local time long enough to judge this step."""
    joint = state.host("way_joint")
    living = joint >= 0
    way_a, way_b, joint = state.host("way_a")[living], state.host("way_b")[living], joint[living]
    rate = state.host("rate").astype(np.float64)
    slowest = np.minimum(np.minimum(rate[way_a], rate[way_b]), rate[joint])
    bar = _STILL_BAR * np.minimum(quanta[way_a], quanta[way_b]) * step_time * slowest
    return int(((quanta[joint] > 0) & (bar >= 1.0)).sum())


def _share(part, whole) -> float:
    """`part / whole` as a float; 0.0 of nothing."""
    return float(part) / float(whole) if whole else 0.0


def check_step(state, settings) -> dict:
    """Axiom 1's counts for this step: the books against the law, wrong boxes, cells standing still."""
    if state.start_books is None:
        raise ValueError("the start's books were never read: state.start_books is None")
    books = read_books(state)
    off = book_law(state.start_books, books)
    quanta = state.host("cell_quanta")
    moved = state.host("moved_inside") + _crossed_by_cell(state, quanta.size)
    bar = _STILL_BAR * quanta * settings.step_time * state.host("rate").astype(np.float64)
    lived = (quanta > 0) & (state.host("clock") > 0.0)
    still = lived & (bar >= 1.0) & (moved < bar)
    counts = {"book_amount": off["amount"], "book_plus": off["plus"], "book_minus": off["minus"],
              "book_taken": off["taken"],
              "still_cells": int(np.count_nonzero(still)),
              "still_share": _share(quanta[still].sum(), quanta.sum()),
              "unjudged_share": _share(quanta[lived & (bar < 1.0)].sum(), quanta.sum()),
              "moved_share": _share(moved.sum() - books["crossed"], quanta.sum()),
              "acting_ways": _acting_ways(state, quanta, settings.step_time),
              "crossed_share": _share(books["crossed"], books["plus"])}
    counts.update({name: books[name] for name in _COPIED})
    return counts
