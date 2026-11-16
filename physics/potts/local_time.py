"""Local time: every cell's own clock advances by the proper time it lived this step.

Contract: `local_time(universe, settings)` adds cell_clock * step_time to cell_time for
every living cell holding boxes, on the GPU (bible: time is local; each region's clock
runs at a rate set by its own energy concentration, the more packed the slower). The
clock is the one the step's energy flow ran on (cell_sums.py, read before the flow
changed the sums), so the reading is the proper time the cell's fluid was advanced.
universe.cell_time is the reading (float64, per cell id). Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.potts.settings import Settings
from physics.potts.universe import Universe


@wp.kernel
def _advance(alive: wp.array(dtype=wp.int32), cell_boxes: wp.array(dtype=wp.int32),
             cell_clock: wp.array(dtype=wp.float32), step_time: float, cell_time: wp.array(dtype=wp.float64)):
    cell = wp.tid()
    if alive[cell] == 0 or cell_boxes[cell] <= 0:
        return
    cell_time[cell] = cell_time[cell] + wp.float64(cell_clock[cell] * step_time)


def local_time(universe: Universe, settings: Settings) -> None:
    """Advance each living cell's local clock by its clock rate times the step."""
    wp.launch(_advance, dim=universe.cell_count, device=universe.owner.device,
              inputs=[universe.alive, universe.cell_boxes, universe.cell_clock, settings.step_time,
                      universe.cell_time])
