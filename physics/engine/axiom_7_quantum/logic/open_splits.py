"""Open splits: a volume's outflow that met a fork is one energy on several routes at once.

Contract: `open_splits(state, rows, claimed)` runs after `cross` and `carry_on`. `rows`
is what each volume sent into each other volume this step (`read_sent`) and `claimed`
how many of those quanta were halves of open splits going on (`carry_on`). It writes
only the store. Nothing is named or staged: it reads what the fluid did.
  New moving energy is what a volume sent that was no open split's: `sent - claimed`
  per crossing (never below 0), so no quantum is ever claimed twice.
  A fork: a volume whose new moving energy went into two or more other volumes this
  step. It is one moving energy, and its halves lie in the volumes that took it.
  A volume with no open split gets one: `born` this step, `directed` what the joints'
  friction left with its direction (each crossing's `new * (sent - lost) // sent`), the
  pointer on one of the halves' cells by share (`pointer_law`, draw `SPLIT_POINTER`
  number 0); counted `opened`.
  A volume whose split is still open sends more of the same energy down the same fork:
  the halves and `directed` grow; counted `joined`.
  New outflow into one volume only met no fork and opens nothing.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_rows
from physics.engine.axiom_1_energy.definitions.passes import SPLIT_POINTER
from physics.engine.axiom_7_quantum.definitions.pointer_law import pointer_law
from physics.engine.axiom_7_quantum.definitions.split_store import PART_COLUMNS, SPLIT_COLUMNS
from physics.engine.axiom_7_quantum.logic.set_splits import set_splits

_SOURCE_BITS = 32


def _forked(giver: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    """Which crossings carry new moving energy of a volume that sent some into two or more volumes this step."""
    takers = np.zeros(int(giver.max()) + 1 if len(giver) else 0, dtype=np.int64)
    np.add.at(takers, giver[fresh > 0], 1)
    return (fresh > 0) & (takers[giver] > 1)


def _new_splits(state, giver: np.ndarray, taker: np.ndarray, sent: np.ndarray, directed_sent: np.ndarray) -> dict:
    """One fresh split row per forking volume: born now, its direction, its pointer by share."""
    sources, row_of = np.unique(giver, return_inverse=True)
    directed = np.zeros(len(sources), dtype=np.int64)
    np.add.at(directed, row_of, directed_sent)
    uniform = draw_uniform_rows(state.seed, state.step, SPLIT_POINTER, sources << _SOURCE_BITS)
    return {"source": sources, "born": np.full(len(sources), int(state.step), dtype=np.int64),
            "pointer": pointer_law(giver, taker, sent, sources, uniform), "directed": directed,
            "reacted": np.zeros(len(sources), dtype=bool)}


def open_splits(state, rows: dict, claimed: np.ndarray) -> None:
    """Open a split, or feed the open one, for every volume whose new outflow forked this step."""
    store = state.quantum
    fresh = np.maximum(rows["sent"] - claimed, 0)
    fork = _forked(rows["giver"], fresh)
    if not fork.any():
        return
    giver, taker, sent = rows["giver"][fork], rows["taker"][fork], fresh[fork]
    directed_sent = sent * (rows["sent"][fork] - rows["lost"][fork]) // rows["sent"][fork]
    new = _new_splits(state, giver, taker, sent, directed_sent)
    joined = int(np.count_nonzero(np.isin(new["source"], store["source"])))
    splits = {name: np.concatenate([store[name], new[name]]) for name in SPLIT_COLUMNS + ("reacted",)}
    parts = {name: np.concatenate([store[name], column]) for name, column in zip(PART_COLUMNS, (giver, taker, sent))}
    set_splits(store, splits, parts)
    for name, count in (("opened", len(new["source"]) - joined), ("joined", joined)):
        store["step"][name] += count
        store["total"][name] += count
