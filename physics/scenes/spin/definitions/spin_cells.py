"""Spin cells: which cells the spin scene watches.

Contract: `spin_cells(side, count)` takes the start lattice's cells along an edge and a
count, and returns `(volumes, joints)`, two lists of `count` cell ids each: energy
volume n sits at lattice place (4n, 4n, 4n) and joint n is the face joint after it
along the third axis, at (4n, 4n, 4n + 1). So no two watched volumes neighbour each
other. Ids are the start's, `(i * side + j) * side + k`. Raises ValueError for a count
under 1 or over side // 4.
"""
from __future__ import annotations


def spin_cells(side: int, count: int) -> tuple[list, list]:
    """The ids of `count` energy volumes along the lattice diagonal and the face joint beside each."""
    if not 1 <= count <= side // 4:
        raise ValueError(f"a {side}-place lattice has room for 1 .. {side // 4} watched volumes, got {count}")
    volumes = [(4 * number * side + 4 * number) * side + 4 * number for number in range(count)]
    return volumes, [cell + 1 for cell in volumes]
