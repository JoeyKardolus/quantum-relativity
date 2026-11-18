"""Read books: every box counted in whole numbers, beside the counts of what changed the total.

Contract: `read_books(state)` returns a dict of exact Python ints.
Counted from every box of every id handed out so far, one thread per cell:
  plus, minus   the quanta in boxes above 0, and in boxes below 0 as a size
  amount        plus + minus
  wrong_boxes   boxes of the wrong sign in a living cell (below 0 in an energy volume,
                above 0 in a joint) and boxes that are not 0 in an id that is no cell
  wrong_cells   ids whose `cell_quanta` is not the sum of their boxes
  empty_cells   living cells that hold 0
This step's ledgers (cleared at the start of every step):
  moved_inside, cut_moves   quanta moved between boxes inside cells; moves cut at 0
  crossed, heated           quanta that crossed a joint (both senses counted) and those
                            of them that lost their direction there (friction)
  heat                      quanta held now that have lost their direction (`heat_quanta`)
  made                      minus quanta squeezed joints made this step
The counts kept since the start:
  made_total, taken_joint, taken_volumes, short_takes, short_quanta, torn
`book_law` (definitions) says what `amount`, `plus` and `minus` must be from the start's
books and these counts. It writes nothing.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, NO_CELL

wp.set_module_options({"fuse_fp": False})

_BLOCK = wp.constant(BLOCK)
_NO_CELL = wp.constant(NO_CELL)
_KEPT_COUNTS = ("made_total", "taken_joint", "taken_volumes", "short_takes", "short_quanta", "torn")


@wp.kernel
def _sum_boxes(kind: wp.array(dtype=wp.int32), quanta: wp.array4d(dtype=wp.int32),
               plus: wp.array(dtype=wp.int64), minus: wp.array(dtype=wp.int64),
               wrong: wp.array(dtype=wp.int32)):
    cell = wp.tid()
    sign = kind[cell]
    above = wp.int64(0)
    below = wp.int64(0)
    bad = int(0)
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                held = quanta[cell, across, along, up]
                if held > 0:
                    above += wp.int64(held)
                if held < 0:
                    below -= wp.int64(held)
                if held != 0 and (sign == _NO_CELL or held * sign < 0):
                    bad += 1
    plus[cell] = above
    minus[cell] = below
    wrong[cell] = bad


def _whole_sum(values: np.ndarray) -> int:
    """The exact sum of a host array of whole numbers."""
    return int(values.astype(np.int64).sum())


def read_books(state) -> dict:
    """The exact books: every box summed in whole numbers, this step's ledgers and the kept counts."""
    plus = wp.zeros(state.count, dtype=wp.int64, device=state.device)
    minus = wp.zeros(state.count, dtype=wp.int64, device=state.device)
    wrong = wp.zeros(state.count, dtype=wp.int32, device=state.device)
    wp.launch(_sum_boxes, dim=state.count, inputs=[state.kind, state.quanta, plus, minus, wrong],
              device=state.device)
    plus_of, minus_of = plus.numpy(), minus.numpy()
    held = plus_of + minus_of
    books = {"plus": _whole_sum(plus_of), "minus": _whole_sum(minus_of)}
    books["amount"] = books["plus"] + books["minus"]
    books["wrong_boxes"] = _whole_sum(wrong.numpy())
    books["wrong_cells"] = int(np.count_nonzero(state.host("cell_quanta") != held))
    books["empty_cells"] = int(np.count_nonzero((state.host("kind") != NO_CELL) & (held == 0)))
    books["moved_inside"] = _whole_sum(state.host("moved_inside"))
    books["cut_moves"] = _whole_sum(state.host("cut_moves"))
    books["crossed"] = _whole_sum(state.host("crossed"))
    books["heated"] = _whole_sum(state.host("heated"))
    books["heat"] = _whole_sum(state.host("heat_quanta"))
    books["made"] = _whole_sum(state.host("made"))
    for name in _KEPT_COUNTS:
        books[name] = int(getattr(state, name))
    return books
