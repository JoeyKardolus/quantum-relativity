"""Title line: the window title from a snapshot's stats.

Contract: `title_line(snapshot, frames_per_second)` returns one line: step, cells
(with the cell count's growth since the start), joints, collapses, occupied boxes vs
start, lumps, sealed lumps, the amount of whole quanta, drawn boxes, the mode ("nearest",
"glow" or "slab", from the snapshot's `look`, default nearest) and where the cut is when
the snapshot's `slice_at` is a number ("slice z <centre>", or for the slab
"slab z <first>..<last>", its first and last box layer centres), steps/s, frames/s, plus
"negative <n>" when boxes hold negative energy, "paused" when paused and
"STOPPED: <error>" when the source raised. Stats keys read (the engine's `stats`):
cells, joints, collapses, cell_count_growth, occupied_growth, lumps, sealed_lumps,
amount; a missing key shows as "?".
"""
from __future__ import annotations

import math

from physics.view.box_points import GLOW, SLAB, SLAB_BOXES

_MODES = {GLOW: "glow", SLAB: "slab"}


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
             _amount(stats.get("amount")),
             f"{len(snapshot.positions):,} boxes drawn",
             _MODES.get(getattr(snapshot, "look", None), "nearest"),
             f"{snapshot.steps_per_second:.1f} steps/s",
             f"{frames_per_second:.0f} fps"]
    slice_at = getattr(snapshot, "slice_at", float("nan"))
    if not math.isnan(slice_at):
        parts.insert(len(parts) - 2, _cut(slice_at, getattr(snapshot, "look", None) == SLAB))
    if snapshot.negative:
        parts.append(f"negative {snapshot.negative}")
    if snapshot.paused:
        parts.append("paused")
    if snapshot.error:
        parts.append(f"STOPPED: {snapshot.error}")
    return " | ".join(parts)


def _cut(slice_at: float, slab: bool) -> str:
    if not slab:
        return f"slice z {slice_at:g}"
    half = (SLAB_BOXES - 1.0) / 2.0
    return f"slab z {slice_at - half:g}..{slice_at + half:g}"


def _times(ratio) -> str:
    return f"{ratio:.2f}x" if ratio is not None else "?"


def _amount(total) -> str:
    return f"quanta {total:,}" if total is not None else "quanta ?"
