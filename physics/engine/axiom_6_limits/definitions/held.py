"""Held: the host's copy of the few cells one step's events touch.

Contract: `Held` is a record the events of one step share, so each cell is read from
the device once and written back once.
  cells    the ids touched, in the order they were first touched
  quanta   {id: (6, 6, 6) numpy int32}, read by `Grid.block("quanta", id)`
  phase    {id: (6, 6, 6) numpy float64}, read by `Grid.block("phase", id)`
  entries  {id: neighbour entries [(id, code)]}
  values   {array name: {id: value}}: per-cell values the events of this step set, so a
           later event of the step reads what an earlier one wrote
  whole    {array name: host copy of that whole device array}, read once for a step with
           many events so each cell is not fetched from the device by itself; empty when
           the events are few. A cell is still worked on only through the three above.
It works on only the cells an event touches, never the whole grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Held:
    """The blocks and relations of the cells this step's events touch."""

    cells: list = field(default_factory=list)
    quanta: dict = field(default_factory=dict)
    phase: dict = field(default_factory=dict)
    entries: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    whole: dict = field(default_factory=dict)
