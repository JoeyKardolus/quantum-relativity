"""Cell stats: the engine's counts for the title.

Contract: `cell_stats(state, start_size)` returns a dict with every name of
`STAT_NAMES`, read from the engine's own `measure_grid` (cells by kind, size, endless,
warp), `read_books` (the exact amount) and its counts of collapses and tears.
`size_growth` is the size over `start_size`. It writes nothing.
"""
from __future__ import annotations

from physics.engine import measure_grid, read_books


def cell_stats(state, start_size: float) -> dict:
    """Counts of one moment of the engine's grid."""
    measured = measure_grid(state)
    return {"volumes": measured["volumes"], "joints": measured["joints"],
            "collapses": int(state.collapses), "tears": int(state.tears), "endless": measured["endless"],
            "size": measured["size"], "size_growth": measured["size"] / start_size,
            "amount": read_books(state)["amount"], "warp": measured["warp"]}
