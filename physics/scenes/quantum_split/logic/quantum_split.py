"""Quantum split scene: where does a split energy turn up whole, and how often?

Contract: `quantum_split(cells_per_side=32, wait=20, seed=1, settings=None,
device="cuda", start=None)` builds physics.engine.new_universe with the settings
(default Settings(); `start` puts a candidate's start amounts in) and runs `wait` steps
with every rule on. Nothing is staged: a volume's outflow that meets a fork splits by
itself (axiom 7). It then reads the open splits (the engine's split store), puts a
detector (physics.engine react) in the cell of every split's largest half, takes one
more step and reads where each split settled. Returns a dict (no verdict):
  cells_per_side, wait, seed, settings (dict)
  splits        splits open at the reading; halves the mean number of cells one lies in
  settled       settlings in the detector step; by_reaction and by_momentum their causes
  whole_share   the share of them that settled whole (nothing short); known 1
  judged, unread   settlings whose split was open at the reading, and those not
  found_in_largest      the share of the judged that settled in their largest half's cell
  expected_in_largest   the mean share those largest halves held one step before: what
                        found_in_largest must be when a cell's share is its chance
  books_gap     the largest miss of the engine's book_law at the end (0 is exact)
Raises ValueError for wait < 1.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import Settings, book_law, new_universe, react, read_books, step
from physics.scenes.quantum_split.definitions.landing import landing
from physics.scenes.quantum_split.definitions.largest_half import largest_half
from physics.scenes.run.definitions.start_settings import start_settings

_REACTION = 0


def quantum_split(cells_per_side: int = 32, wait: int = 20, seed: int = 1, settings: Settings | None = None,
                  device: str = "cuda", start: dict | None = None) -> dict:
    """How many open splits settle whole under a detector, and how often in their largest half."""
    if wait < 1:
        raise ValueError(f"a split needs a step to open, got wait {wait}")
    settings = start_settings(settings, start)
    universe = new_universe(settings, cells_per_side, seed, device=device)
    for _ in range(wait):
        step(universe, settings)
    store = universe.quantum
    sources = store["source"].copy()
    cells, shares = largest_half(sources, store["part_source"], store["part_cell"], store["part_quanta"])
    halves = len(store["part_source"]) / max(len(sources), 1)
    for cell in sorted(set(cells[cells >= 0].tolist())):
        react(universe, cell)
    step(universe, settings)
    settled = universe.quantum["settled"]
    landed = landing(settled["source"], settled["cell"], sources, cells, shares)
    count = len(settled["source"])
    books = read_books(universe)
    return {"cells_per_side": cells_per_side, "wait": wait, "seed": seed, "settings": asdict(settings),
            "splits": int(len(sources)), "halves": float(halves), "settled": int(count),
            "by_reaction": int(np.sum(settled["cause"] == _REACTION)),
            "by_momentum": int(np.sum(settled["cause"] != _REACTION)),
            "whole_share": float(np.mean(settled["short"] == 0)) if count else 0.0,
            "judged": landed["judged"], "unread": landed["unread"], "found_in_largest": landed["found"],
            "expected_in_largest": landed["expected"],
            "books_gap": max(abs(gap) for gap in book_law(universe.start_books, books).values())}
