"""Title line: the window title from a snapshot's stats.

Contract: `title_line(snapshot, frames_per_second)` returns one line: step, energy
volumes, joints, collapses, tears, lumps, the size of the universe (the sum of its
cells' lengths) with its growth since the start, the quanta, cells drawn, the view
("slice z <layer>" or "whole"), steps/s, frames/s, plus "paused" when paused and
"STOPPED: <error>" when the source raised. Stats keys read (the engine's `stats`):
volumes, joints, collapses, tears, lumps, size, size_growth, amount; a missing key
shows as "?".
"""
from __future__ import annotations


def title_line(snapshot, frames_per_second: float) -> str:
    """One title line from the stats, rates and state of the stepping."""
    stats = snapshot.stats
    size, growth, amount = stats.get("size"), stats.get("size_growth"), stats.get("amount")
    parts = [f"step {snapshot.step}",
             f"volumes {stats.get('volumes', '?')}",
             f"joints {stats.get('joints', '?')}",
             f"collapses {stats.get('collapses', '?')}",
             f"tears {stats.get('tears', '?')}",
             f"lumps {stats.get('lumps', '?')}",
             "size ?" if size is None else f"size {size:,.0f}" + ("" if growth is None else f" ({growth:.4f}x)"),
             "quanta ?" if amount is None else f"quanta {amount:,}",
             f"{snapshot.drawn:,} cells drawn",
             "whole" if snapshot.layer is None else f"slice z {snapshot.layer}",
             f"{snapshot.steps_per_second:.1f} steps/s",
             f"{frames_per_second:.0f} fps"]
    if snapshot.paused:
        parts.append("paused")
    if snapshot.error:
        parts.append(f"STOPPED: {snapshot.error}")
    return " | ".join(parts)
