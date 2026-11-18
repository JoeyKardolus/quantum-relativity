"""Split store: the splits that are open now, as host tables of whole numbers.

Contract: `no_rows()` is an empty int64 column. `empty_store()` returns the store `start_store` puts on `state.quantum`.
A split is the energy one volume's fluid sent out through more than one way (a fork):
it lies in several cells at once. The quanta stay in the boxes; the store only says how
many quanta of which cell are whose, and no quantum is claimed by two splits: a half
goes on with its cell's fluid (`carry_on`), and only outflow that is no open split's
opens a new one.
  One row per open split, in rising `source` order:
    source    the volume whose outflow split
    born      the step it split at
    pointer   the cell its pointer is on, one of its halves' cells
    directed  how many of its quanta still have their direction (0: no momentum left);
              friction lowers it at every crossing its halves make (`carry_on`)
    reacted   whether something has reacted with one of its halves
  One row per half, in rising (`part_source`, `part_cell`) order:
    part_source, part_cell, part_quanta   the split, the cell, the whole quanta there
  detected    the cells in which a caller's detector reacts at the next settling
  step        this step's counts (`STEP_COUNTS`), started afresh by `meet_fork`
  total       the same counts since the start
  settled     this step's settlings, one row each: `source`, `cell` (where it settled
              whole), `amount`, `short` and `cause` (0 reaction, 1 momentum)
Every table is int64 (bool for `reacted`): the store holds no fraction, so it holds no
speed and no turn count. A split's speed is worked again from its own draw each time it
turns (`turn_splits`) and is written nowhere.
`STEP_COUNTS`: opened, joined (forked outflow that joined an open split), carried (quanta of
halves that went on with their cell's fluid), cooled (direction friction took off open
splits), settled_reaction,
settled_momentum, settled_quanta (quanta of the splits that settled), drawn_quanta (of
those, drawn from another cell into the pointer's), short_quanta (not there to be
drawn), torn_quanta (turned minus by a tear), ended (splits left holding nothing).
`amount_of(store)` returns each open split's quanta, the sum of its halves.
"""
from __future__ import annotations

import numpy as np

SPLIT_COLUMNS = ("source", "born", "pointer", "directed")
PART_COLUMNS = ("part_source", "part_cell", "part_quanta")
SETTLED_COLUMNS = ("source", "cell", "amount", "short", "cause")
STEP_COUNTS = ("opened", "joined", "carried", "cooled", "settled_reaction", "settled_momentum", "settled_quanta", "drawn_quanta",
               "short_quanta", "torn_quanta", "ended")
REACTION = 0
MOMENTUM = 1


def no_rows() -> np.ndarray:
    """An empty column of whole numbers."""
    return np.zeros(0, dtype=np.int64)


def empty_store() -> dict:
    """A store in which no split is open and nothing has happened."""
    store = {name: no_rows() for name in SPLIT_COLUMNS + PART_COLUMNS}
    store["reacted"] = np.zeros(0, dtype=bool)
    store["detected"] = set()
    store["step"] = dict.fromkeys(STEP_COUNTS, 0)
    store["total"] = dict.fromkeys(STEP_COUNTS, 0)
    store["settled"] = {name: no_rows() for name in SETTLED_COLUMNS}
    return store


def amount_of(store: dict) -> np.ndarray:
    """The whole quanta of each open split: the sum of its halves."""
    amount = np.zeros(len(store["source"]), dtype=np.int64)
    np.add.at(amount, np.searchsorted(store["source"], store["part_source"]), store["part_quanta"])
    return amount
