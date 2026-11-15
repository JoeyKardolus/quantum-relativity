"""Joint swing: each joint is a spring with inertia, so its size overshoots and keeps swinging (rule 5).

Contract: `joint_swing(universe, settings)` sets, on the GPU, every living cell's aim,
the box count its box moves aim at this step (box_moves.py), and remembers its box
count in previous_boxes:
  an energy volume aims at its want: its size follows its energy and it does not
  vibrate on its own (rule 5);
  a joint aims where a spring would carry it, Stormer-Verlet in its own time
  tau = step_time * clock: aim = n + (n - n_before) - omega**2 * tau**2 * (n - want),
  omega = pi * sound / n**(1/3) (the fluid's slowest standing wave across the joint),
  clipped to at least 1. A joint pressed smaller keeps moving the way it was pressed,
  turns, overshoots its balance and swings back: joints are never perfectly still.
  Its velocity is what its size actually did over the last step, so a joint that could
  not follow its aim (move limits, neighbours) loses that part of its swing, and the
  swing never runs away. Its swing changes its compression and so its seams' leak.
Returns None.
"""
from __future__ import annotations

import math

import warp as wp

from physics.potts.settings import Settings
from physics.potts.universe import Universe


@wp.kernel
def _swing(kind: wp.array(dtype=wp.int32), cell_boxes: wp.array(dtype=wp.int32), cell_want: wp.array(dtype=wp.float32),
           cell_clock: wp.array(dtype=wp.float32), step_time: float, wave: float,
           previous_boxes: wp.array(dtype=wp.float32), cell_aim: wp.array(dtype=wp.float32)):
    cell = wp.tid()
    boxes = float(cell_boxes[cell])
    if boxes <= 0.0:
        cell_aim[cell] = 0.0
        previous_boxes[cell] = 0.0
        return
    want = cell_want[cell]
    aim = want
    if kind[cell] == 1:
        omega = wave / wp.pow(boxes, 1.0 / 3.0)
        tau = step_time * cell_clock[cell]
        before = previous_boxes[cell]
        if before <= 0.0:
            before = boxes
        aim = wp.max(1.0, boxes + (boxes - before) - omega * omega * tau * tau * (boxes - want))
    cell_aim[cell] = aim
    previous_boxes[cell] = boxes


def joint_swing(universe: Universe, settings: Settings) -> None:
    """Set each cell's aim: its want, or for a joint the next size of its swing."""
    wp.launch(_swing, dim=universe.cell_count, device=universe.owner.device,
              inputs=[universe.kind, universe.cell_boxes, universe.cell_want, universe.cell_clock, settings.step_time,
                      math.pi * settings.sound, universe.previous_boxes, universe.cell_aim])
