"""Size ladder: the sizes a search may try, in cells a side.

`SIZES` is every even side from 4 (the engine's smallest lattice; a side must be even
for the A B A B pattern to wrap) to 64, chosen before the search. `check_sizes(sizes)`
raises ValueError for a list that is empty or not strictly rising: the search bisects,
which needs an order.
"""
from __future__ import annotations

from typing import Sequence

SIZES = tuple(range(4, 65, 2))


def check_sizes(sizes: Sequence[int]) -> None:
    """Refuse a size list the bisection cannot search."""
    ordered = list(sizes)
    if not ordered or any(later <= earlier for earlier, later in zip(ordered, ordered[1:])):
        raise ValueError(f"need a non-empty, strictly rising size list, got {ordered}")
