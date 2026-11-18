"""Expected turns: the winding a cell of the engine reads when it is one whole swirl.

Contract: `expected_turns(axis, sense)` takes a cell's `swirl_axis` (0, 1 or 2) and
`swirl_sense` (+1 or -1) as the engine holds them (`state.host("swirl_axis")`,
`state.host("swirl_sense")`) and returns the whole turns `cell_turns` reads in every
layer across that axis while the cell is the swirl axiom 2 says it is:
  axis 0 or 2   -sense
  axis 1        +sense
Why the signs: the engine winds the phase by minus its sense, measured from the first
box coordinate across the axis toward the second in rising order (axis 1: x toward z);
this package walks a square right-handed (in_plane.py; axis 1: z toward x), which is
the other way round for axis 1 only. Raises ValueError for any other axis or sense.
"""
from __future__ import annotations

from physics.measure.winding.definitions.in_plane import AXES

_LEFT_HANDED_AXIS = 1


def expected_turns(axis: int, sense: int) -> int:
    """The turns per layer of a cell that is one whole swirl round `axis` in the sense `sense`."""
    if axis not in range(AXES) or sense not in (-1, 1):
        raise ValueError(f"need axis 0, 1 or 2 and sense +1 or -1, got {axis} and {sense}")
    return int(sense) if axis == _LEFT_HANDED_AXIS else -int(sense)
