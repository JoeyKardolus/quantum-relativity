"""Read cells: the per-cell numbers a viewer draws, as host arrays.

Contract: `read_cells(state)` returns `{name: numpy array}` with one row per id handed
out so far, for `kind`, `length`, `cell_quanta`, `pull`, `rate`, `fell_into` and
`torn_from`. Copies: changing them changes nothing in the state. It writes nothing.
"""
from __future__ import annotations

_NAMES = ("kind", "length", "cell_quanta", "pull", "rate", "fell_into", "torn_from")


def read_cells(state) -> dict:
    """Host copies of the per-cell arrays the viewer reads."""
    return {name: state.host(name) for name in _NAMES}
