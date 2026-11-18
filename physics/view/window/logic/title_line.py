"""Title line: the window title from a snapshot.

Contract: `title_line(snapshot, frames_per_second)` returns one line: the step, every
count of `STAT_NAMES` (size with its growth since the start in brackets; a missing
count shows as "?"), cells drawn, the view ("whole" or "slice z <layer>"), steps/s,
frames/s, plus "paused" when paused and "STOPPED: <error>" when the source raised.
"""
from __future__ import annotations


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
              "whole" if snapshot.layer is None else f"slice z {snapshot.layer}",
              f"{snapshot.steps_per_second:.1f} steps/s",
              f"{frames_per_second:.0f} fps"]
    if snapshot.paused:
        parts.append("paused")
    if snapshot.error:
        parts.append(f"STOPPED: {snapshot.error}")
    return " | ".join(parts)
