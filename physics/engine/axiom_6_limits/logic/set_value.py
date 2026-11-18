"""Set value: an event writes one cell's value of a per-cell array.

Contract: `set_value(state, held, name, cell, value)` writes it on the grid
(`Grid.set_cell`) and remembers it in `held.values`, so the later events of the same
step read it with `cell_value` without going back to the device.
"""
from __future__ import annotations


def set_value(state, held, name: str, cell: int, value) -> None:
    """Write one cell's value of `name` and remember it for this step's later events."""
    state.set_cell(name, int(cell), value)
    held.values.setdefault(name, {})[int(cell)] = value
