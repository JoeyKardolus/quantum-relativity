"""Hold cell: bring one cell an event touches into the step's `Held` record, once.

Contract: `hold_cell(state, held, cell, with_block=False)`. The first time a cell is
named, its id is appended to `held.cells` and its neighbour entries are read into
`held.entries[cell]`. With `with_block`, its blocks of `quanta` and `phase` are read
into `held.quanta[cell]` and `held.phase[cell]`, also once. A cell already held is left
as the earlier events of this step made it, so every event of a step sees the others'
changes and each cell is read from the device once (or, in a step with many events, out
of `held.whole`, the whole arrays read once). The grid itself is not changed.
"""
from __future__ import annotations


def _block(state, held, name: str, cell: int):
    """One cell's block: out of the step's whole host copy when there is one, else from the device."""
    if name in held.whole:
        return held.whole[name][cell].copy()
    return state.block(name, cell)


def hold_cell(state, held, cell: int, with_block: bool = False) -> None:
    """Read a cell's relations (and, when asked, its blocks) into `held` unless they are there."""
    cell = int(cell)
    if cell not in held.entries:
        held.cells.append(cell)
        held.entries[cell] = state.neighbours(cell)
    if with_block and cell not in held.quanta:
        held.quanta[cell] = _block(state, held, "quanta", cell)
        held.phase[cell] = _block(state, held, "phase", cell)
