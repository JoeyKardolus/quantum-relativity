"""Check step: axiom 7's counts after one step of a run, for the per-step records.

Contract: `check_step(state, settings) -> dict`, read after a whole step. It changes nothing.
  open              splits open now
  opened, joined, carried, cooled, settled_reaction, settled_momentum, settled_quanta,
  drawn_quanta, short_quanta, torn_quanta, ended    this step's counts (split_store.py)
  claimed, directed  quanta the open splits claim now, and how many still have their direction
  over_claimed      must be 0: quanta the open splits claim in cells beyond what those
                    cells hold (no quantum is claimed twice)
  parts_wrong       must be 0: open splits with no half, a half of 0 or less, a half in
                    a cell that is no living energy volume, a `directed` outside 0 to
                    the split's quanta, a pointer on a cell that holds no half of it, or
                    a `born` after this step; and tables out of order
  not_whole         must be 0: settlings of this step that came short (some of the
                    split's quanta were not there to be drawn), or that settled in a cell
                    that is no living energy volume
  fractions         must be 0: tables of the store that are not whole numbers (7e: the
                    store can hold no speed and no turn count)
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME
from physics.engine.axiom_7_quantum.definitions.split_store import STEP_COUNTS, amount_of

_SOURCE_BITS = 32


def _parts_wrong(state, store: dict) -> int:
    """How many open splits, or tables, break what the store must keep."""
    kind = state.host("kind")
    key = (store["part_source"] << _SOURCE_BITS) | store["part_cell"]
    amount = amount_of(store)
    wrong = (amount <= 0) | (store["directed"] < 0) | (store["directed"] > amount) | (store["born"] > state.step)
    wrong |= ~np.isin((store["source"] << _SOURCE_BITS) | store["pointer"], key)
    bad_half = (store["part_quanta"] <= 0) | (kind[store["part_cell"]] != ENERGY_VOLUME)
    wrong[np.searchsorted(store["source"], store["part_source"][bad_half])] = True
    unordered = int(np.any(np.diff(store["source"]) <= 0)) + int(np.any(np.diff(key) <= 0))
    return int(np.count_nonzero(wrong)) + unordered


def _not_whole(state, store: dict) -> int:
    """How many of this step's settlings did not end as one whole amount in one living volume."""
    settled = store["settled"]
    kind = state.host("kind")
    return int(np.count_nonzero((settled["amount"] < 0) | (settled["short"] != 0)
                                | (kind[settled["cell"]] != ENERGY_VOLUME)))


def _over_claimed(state, store: dict) -> int:
    """Quanta the open splits claim in cells beyond what those cells hold."""
    claims = np.zeros(len(state.host("cell_quanta")), dtype=np.int64)
    np.add.at(claims, store["part_cell"], store["part_quanta"])
    return int(np.maximum(claims - state.host("cell_quanta"), 0).sum())


def check_step(state, settings) -> dict:
    """Axiom 7's counts for the state as it stands after a step."""
    store = state.quantum
    tables = [column for column in list(store.values()) + list(store["settled"].values())
              if isinstance(column, np.ndarray)]
    counts = {name: int(store["step"][name]) for name in STEP_COUNTS}
    counts.update({
        "open": int(len(store["source"])),
        "claimed": int(store["part_quanta"].sum()),
        "directed": int(store["directed"].sum()),
        "over_claimed": _over_claimed(state, store),
        "parts_wrong": _parts_wrong(state, store),
        "not_whole": _not_whole(state, store),
        "fractions": sum(column.dtype not in (np.int64, np.bool_) for column in tables)
        + sum(not isinstance(value, int) for value in list(store["step"].values()) + list(store["total"].values())),
    })
    return counts
