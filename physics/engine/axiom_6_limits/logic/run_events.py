"""Run events: carry out every collapse and tear `find_limits` marked in this step.

Contract: `run_events(state, settings)` works on the host, on only the cells the events
touch (`Held`), collapses first, each kind in id order.
  Every flagged joint that still lives collapses, also when an earlier event of the step
  already touched it: a joint past its limit is never left standing.
  Every flagged volume that still lives and still falls under `tear_law` (an earlier
  collapse may have merged it, so it is counted again from its own boxes) tears, with
  two new ids.
Then the blocks and relations are written back, `version` rises by the number of events
and one record per event is appended to `state.changed`. A tear's record gets
`"lengths"`, the three pieces' lengths by the size law. Returns the ids of every cell
whose relations it wrote: the caller hands them to `relink` (axiom 3), which rebuilds
the ways. Until then the way table is stale.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_3_structure.definitions.size_law import size_law_host
from physics.engine.axiom_6_limits.definitions.held import Held
from physics.engine.axiom_6_limits.definitions.limits_arrays import COLLAPSE_FLAG, TEAR_FLAG
from physics.engine.axiom_6_limits.logic.cell_value import cell_value
from physics.engine.axiom_6_limits.logic.collapse import collapse
from physics.engine.axiom_6_limits.logic.hold_cell import hold_cell
from physics.engine.axiom_6_limits.logic.tear import tear

_FEW_EVENTS = 8
_WHOLE = ("quanta", "phase", "kind", "squeezed_way", "way_press", "pull", "pull_old", "heat_quanta",
          "swirl_axis", "swirl_sense")
_PIECE = 2


def _reserve(state, volumes: list) -> None:
    """Make room, in one `grow`, for the cells and ways every flagged tear can add."""
    ways = sum(2 * len(state.neighbours(volume)) + 9 for volume in volumes)
    state.grow(2 * len(volumes), ways)


def _read_whole(state, held, events: int) -> None:
    """In a step with many events, read the arrays the events need whole, once, into `held.whole`."""
    if events > _FEW_EVENTS:
        for name in _WHOLE:
            held.whole[name] = getattr(state, name).numpy().copy()


def _still_tears(state, held, settings, volume: int) -> bool:
    """Whether a flagged volume still lives and still is too long with three pieces that hold something."""
    if int(cell_value(state, held, "kind", volume)) != ENERGY_VOLUME:
        return False
    hold_cell(state, held, volume, with_block=True)
    layers = np.moveaxis(held.quanta[volume].astype(np.int64), int(cell_value(state, held, "swirl_axis", volume)), 0)
    pieces = [int(layers[first:first + _PIECE].sum()) for first in range(0, len(layers), _PIECE)]
    return min(pieces) > 0 and size_law_host(sum(pieces), settings.size_constant) > settings.tear_length


def _write_back(state, held) -> None:
    """Put the held blocks and relations of every touched cell back on the grid."""
    for cell in held.cells:
        state.set_neighbours(cell, held.entries[cell])
    if "quanta" in held.whole:
        for name, blocks in (("quanta", held.quanta), ("phase", held.phase)):
            for cell, block in blocks.items():
                held.whole[name][cell] = block
            getattr(state, name).assign(held.whole[name])
        return
    for cell in held.quanta:
        state.set_block("quanta", cell, held.quanta[cell])
        state.set_block("phase", cell, held.phase[cell])


def run_events(state, settings) -> list:
    """Carry out every flagged collapse and tear of this step; returns the cells whose relations changed."""
    flags = state.host("flags")
    joints = [int(cell) for cell in np.flatnonzero(flags == COLLAPSE_FLAG)]
    volumes = [int(cell) for cell in np.flatnonzero(flags == TEAR_FLAG)]
    _reserve(state, volumes)
    held, records = Held(), []
    _read_whole(state, held, len(joints) + len(volumes))
    for joint in joints:
        if int(cell_value(state, held, "kind", joint)) == JOINT:
            records.append(collapse(state, held, joint))
    for volume in volumes:
        if _still_tears(state, held, settings, volume):
            new_joint, new_volume = state.new_ids(2)
            record = tear(state, held, volume, new_joint, new_volume)
            record["lengths"] = [size_law_host(quanta, settings.size_constant) for quanta in record["quanta"]]
            records.append(record)
    _write_back(state, held)
    state.version += len(records)
    state.changed.extend(records)
    return list(held.cells)
