"""Range rule: how far the shells and the walks of a dimension measure may go in a graph.

Contract: `range_rule(farthest)` takes the largest number of steps from the first source
to any node it can reach and returns (radii, times), two int64 arrays:
  reach   max(3, farthest // 4): a quarter of the way to the farthest node, so a ball
          and a walk stay inside the graph and never meet its end or themselves
  radii   2 .. reach, the shell radii for the ball dimension
  times   max(2, reach**2 // 10) .. reach**2, the step counts for the spectral
          dimension (a walk of t steps spreads about sqrt(t) far)
The same rule is used on the reference grid and on the engine's grid, so the two
numbers can be set side by side. Raises ValueError for farthest < 1. In a graph smaller
than 12 steps across the reach of 3 is more than a quarter; the measure then says so by
its own number, or raises on an empty shell.
"""
from __future__ import annotations

import numpy as np

SHARE_OF_FARTHEST = 4
SMALLEST_REACH = 3
FIRST_RADIUS = 2
SHORTEST_WALK = 2
WALK_SHARE = 10


def range_rule(farthest: int) -> tuple[np.ndarray, np.ndarray]:
    """The shell radii and the walk times for a graph whose farthest node is `farthest` steps off."""
    if farthest < 1:
        raise ValueError(f"the farthest node must be at least 1 step off, got {farthest}")
    reach = max(SMALLEST_REACH, int(farthest) // SHARE_OF_FARTHEST)
    radii = np.arange(FIRST_RADIUS, reach + 1, dtype=np.int64)
    times = np.arange(max(SHORTEST_WALK, reach ** 2 // WALK_SHARE), reach ** 2 + 1, dtype=np.int64)
    return radii, times
