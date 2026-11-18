"""Set splits: write the store's tables back in order, merged, with nothing empty left in them.

Contract: `set_splits(store, splits, parts)` takes the open splits as a dict of equal
long columns (`source`, `born`, `pointer`, `directed`, `reacted`) and the halves as
(`part_source`, `part_cell`, `part_quanta`), in any order and with repeats, and writes
the store's tables:
  halves of one split in one cell are added up; a half of 0 is dropped;
  rows of one source become one split: the earliest `born`, the `directed` added up, the
  first row's `pointer`, `reacted` if any was; `directed` is never more than its quanta;
  a split left with no half is dropped. Returns how many were.
  Both tables come out in rising order (split_store.py).
It changes nothing else of the store and reads no array of the state.
"""
from __future__ import annotations

import numpy as np

_SOURCE_BITS = 32


def _merged_parts(parts: dict) -> dict:
    """The halves with one row per (split, cell) that holds quanta, in rising order."""
    key = (parts["part_source"].astype(np.int64) << _SOURCE_BITS) | parts["part_cell"].astype(np.int64)
    keys, row_of = np.unique(key, return_inverse=True)
    quanta = np.zeros(len(keys), dtype=np.int64)
    np.add.at(quanta, row_of, parts["part_quanta"])
    keys, quanta = keys[quanta > 0], quanta[quanta > 0]
    return {"part_source": keys >> _SOURCE_BITS, "part_cell": keys & ((1 << _SOURCE_BITS) - 1),
            "part_quanta": quanta}


def _merged_splits(splits: dict) -> dict:
    """One row per source, in rising order: earliest born, directed added, first pointer, any reaction."""
    sources, first, row_of = np.unique(splits["source"], return_index=True, return_inverse=True)
    born = np.full(len(sources), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(born, row_of, splits["born"])
    directed = np.zeros(len(sources), dtype=np.int64)
    np.add.at(directed, row_of, splits["directed"])
    reacted = np.zeros(len(sources), dtype=bool)
    np.logical_or.at(reacted, row_of, splits["reacted"])
    return {"source": sources.astype(np.int64), "born": born, "pointer": splits["pointer"][first].astype(np.int64),
            "directed": directed, "reacted": reacted}


def set_splits(store: dict, splits: dict, parts: dict) -> int:
    """Write the merged, ordered tables into the store; returns the splits dropped for holding nothing."""
    parts = _merged_parts(parts)
    splits = _merged_splits(splits)
    amount = np.zeros(len(splits["source"]), dtype=np.int64)
    place = np.searchsorted(splits["source"], parts["part_source"])
    np.add.at(amount, place, parts["part_quanta"])
    holds = amount > 0
    for name, column in splits.items():
        store[name] = column[holds]
    store["directed"] = np.minimum(store["directed"], amount[holds])
    store.update(parts)
    return int(np.count_nonzero(~holds))
