"""Title line: the window title from a snapshot's stats.

Contract: `title_line(snapshot, frames_per_second)` returns one line: step, lumps,
sealed lumps, collapsed cores, size vs start, total energy, drawn boxes, steps/s,
frames/s, plus "negative <n>" when boxes hold negative energy, "paused" when
paused and "STOPPED: <error>" when the source raised. Stats keys read: lumps,
sealed_lumps, collapsed_cores, size_vs_start, total_energy; a missing key shows
as "?".
"""
from __future__ import annotations


def title_line(snapshot, frames_per_second: float) -> str:
    """One title line from the stats, rates and state of the stepping."""
    stats = snapshot.stats
    size = stats.get("size_vs_start")
    total = stats.get("total_energy")
    parts = [f"step {snapshot.step}",
             f"lumps {stats.get('lumps', '?')}",
             f"sealed {stats.get('sealed_lumps', '?')}",
             f"collapses {stats.get('collapsed_cores', '?')}",
             f"size {size:.2f}x start" if size is not None else "size ?",
             f"energy {total:.9g}" if total is not None else "energy ?",
             f"{len(snapshot.positions):,} boxes drawn",
             f"{snapshot.steps_per_second:.1f} steps/s",
             f"{frames_per_second:.0f} fps"]
    if snapshot.negative:
        parts.append(f"negative {snapshot.negative}")
    if snapshot.paused:
        parts.append("paused")
    if snapshot.error:
        parts.append(f"STOPPED: {snapshot.error}")
    return " | ".join(parts)
