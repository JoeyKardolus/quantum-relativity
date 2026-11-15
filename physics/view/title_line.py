"""Title line: the window title from a snapshot's stats.

Contract: `title_line(snapshot, frames_per_second)` returns one line: step, volumes,
collapses, splits, size vs start, lumps, sealed lumps, steps/s, frames/s, plus
"paused" when paused and "STOPPED: <error>" when the source raised. Stats keys
read: volumes, collapses, splits, size_vs_start, lumps, sealed_lumps; a missing
key shows as "?".
"""
from __future__ import annotations


def title_line(snapshot, frames_per_second: float) -> str:
    """One title line from the stats, rates and state of the stepping."""
    stats = snapshot.stats
    size = stats.get("size_vs_start")
    parts = [f"step {snapshot.step}",
             f"volumes {stats.get('volumes', '?')}",
             f"collapses {stats.get('collapses', '?')}",
             f"splits {stats.get('splits', '?')}",
             f"size {size:.2f}x start" if size is not None else "size ?",
             f"lumps {stats.get('lumps', '?')}",
             f"sealed {stats.get('sealed_lumps', '?')}",
             f"{snapshot.steps_per_second:.1f} steps/s",
             f"{frames_per_second:.0f} fps"]
    if snapshot.paused:
        parts.append("paused")
    if snapshot.error:
        parts.append(f"STOPPED: {snapshot.error}")
    return " | ".join(parts)
