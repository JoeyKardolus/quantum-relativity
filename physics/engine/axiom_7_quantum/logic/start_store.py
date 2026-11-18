"""Start store: give a new universe its empty store of splits.

Contract: `start_store(state)` sets `state.quantum` to an empty store (split_store.py).
It reads nothing and writes nothing else. No split is open until some volume's outflow
meets a fork (`open_splits`).
"""
from __future__ import annotations

from physics.engine.axiom_7_quantum.definitions.split_store import empty_store


def start_store(state) -> None:
    """Put an empty store of splits on the state."""
    state.quantum = empty_store()
