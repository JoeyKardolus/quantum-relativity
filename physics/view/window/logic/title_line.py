"""Title line: the window title from a snapshot.

Contract: `title_line(snapshot, frames_per_second)` returns one line: the step, every
count of `STAT_NAMES` (size with its growth since the start in brackets; a missing
count shows as "?"), cells drawn, the view ("slice z <layer>", after "whole + " when
the packed cells' blocks show too), steps/s, frames/s, plus "sizes drawn up to <LIMIT>x"
when a cell's size is further from the typical than is drawn, "<n> packed cells too
spread for blocks" when the whole grid left them out, "paused" when paused and
"STOPPED: <error>" when the source raised.
"""
from __future__ import annotations

from physics.view.layout.definitions.warp_marks import LIMIT


def title_line(snapshot, frames_per_second: float) -> str:
    """One title line from the stats, rates and state of the stepping."""
    stats = snapshot.stats
    size, growth, amount, warp = (stats.get(name) for name in ("size", "size_growth", "amount", "warp"))
    parts = [f"step {snapshot.step}"]
    parts += [f"{name} {stats.get(name, '?')}" for name in ("volumes", "joints", "collapses", "tears", "endless")]
    parts += ["size ?" if size is None else f"size {size:,.0f}" + ("" if growth is None else f" ({growth:.4f}x)"),
              "quanta ?" if amount is None else f"quanta {amount:,}",
              "warp ?" if warp is None else f"warp {warp:.3f}",
              f"{snapshot.drawn:,} cells drawn",
              ("" if snapshot.blocks is None else "whole + ") + f"slice z {snapshot.layer}",
              f"{snapshot.steps_per_second:.1f} steps/s",
              f"{frames_per_second:.0f} fps"]
    if snapshot.limited:
        parts.append(f"sizes drawn up to {LIMIT:g}x")
    if snapshot.left_out:
        parts.append(f"{snapshot.left_out:,} packed cells too spread for blocks")
    if snapshot.paused:
        parts.append("paused")
    if snapshot.error:
        parts.append(f"STOPPED: {snapshot.error}")
    return " | ".join(parts)
