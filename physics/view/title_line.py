"""Title line: the window title from a snapshot's stats.

Contract: `title_line(snapshot, frames_per_second)` returns one line: step, cells
(with the cell count's growth since the start), joints, collapses, occupied boxes vs
start, lumps, sealed lumps, total energy, drawn boxes, the look ("nearest" or
"glow", from the snapshot's `look`, default nearest) and "slice z <centre>" when
the snapshot's `slice_at` is a number, steps/s, frames/s, plus
"negative <n>" when boxes hold negative energy, "paused" when paused and
"STOPPED: <error>" when the source raised. Stats keys read (the engine's `stats`):
cells, joints, collapses, cell_count_growth, occupied_growth, lumps, sealed_lumps,
total_energy; a missing key shows as "?".
"""
from __future__ import annotations

import math

from physics.view.box_points import GLOW


def title_line(snapshot, frames_per_second: float) -> str:
    """One title line from the stats, rates and state of the stepping."""
    stats = snapshot.stats
    parts = [f"step {snapshot.step}",
             f"cells {stats.get('cells', '?')} ({_times(stats.get('cell_count_growth'))})",
             f"joints {stats.get('joints', '?')}",
             f"collapses {stats.get('collapses', '?')}",
             f"occupied {_times(stats.get('occupied_growth'))} start",
             f"lumps {stats.get('lumps', '?')}",
             f"sealed {stats.get('sealed_lumps', '?')}",
             _energy(stats.get("total_energy")),
             f"{len(snapshot.positions):,} boxes drawn",
             "glow" if getattr(snapshot, "look", None) == GLOW else "nearest",
             f"{snapshot.steps_per_second:.1f} steps/s",
             f"{frames_per_second:.0f} fps"]
    slice_at = getattr(snapshot, "slice_at", float("nan"))
    if not math.isnan(slice_at):
        parts.insert(len(parts) - 2, f"slice z {slice_at:g}")
    if snapshot.negative:
        parts.append(f"negative {snapshot.negative}")
    if snapshot.paused:
        parts.append("paused")
    if snapshot.error:
        parts.append(f"STOPPED: {snapshot.error}")
    return " | ".join(parts)


def _times(ratio) -> str:
    return f"{ratio:.2f}x" if ratio is not None else "?"


def _energy(total) -> str:
    return f"energy {total:.9g}" if total is not None else "energy ?"
