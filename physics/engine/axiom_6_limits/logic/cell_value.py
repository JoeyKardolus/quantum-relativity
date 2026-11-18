"""Cell value: one cell's value of a per-cell array, as this step's events have left it.

Contract: `cell_value(state, held, name, cell)` returns the value an earlier event of
this step set (`held.values`), else the step's whole host copy of the array
(`held.whole`), else the value on the device or host. It changes nothing.
"""
from __future__ import annotations


def cell_value(state, held, name: str, cell: int):
    """The value of `name` for one cell, seen through this step's events."""
    cell = int(cell)
    if cell in held.values.get(name, {}):
        return held.values[name][cell]
    if name in held.whole:
        return held.whole[name][cell].item()
    values = getattr(state, name)
    if hasattr(values, "numpy"):
        return values[cell:cell + 1].numpy()[0].item()
    return values[cell].item()
