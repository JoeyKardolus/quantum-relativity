"""Friction law: the joint takes the direction off a share of what crosses it, and that is heat.

Contract: three Warp funcs.
  heat_leaving(moved, heat, holds) -> int64
      of `moved` quanta leaving a cell that holds `holds` quanta, `heat` of them heat:
      `moved * heat // holds` are heat already (the whole part, so a cell never sends
      more heat than it has); 0 for an empty cell
  friction_law(crossing, friction, uniform) -> int64
      of `crossing` quanta going through a joint, `floor(crossing * friction + uniform)`
      lose their direction there, never more than `crossing`. `friction` is the joint's
      share (crossing_settings.py), `uniform` a draw in [0, 1). The joint takes it off
      everything that crosses, heat or not: it cannot tell them apart
  heat_made(crossing, was_heat, lost) -> int64
      of the `lost` quanta, the whole share that was not heat before,
      `lost * (crossing - was_heat) // crossing`: the new heat this crossing made
No quantum is lost: what loses its direction arrives all the same, at rest in the fluid
it joins (arrival_law.py), and is booked in the taking cell's `heat_quanta`. This is the
only friction in the engine.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def heat_leaving(moved: wp.int64, heat: wp.int64, holds: wp.int64) -> wp.int64:
    """How many of the quanta leaving a cell are heat already."""
    if holds <= wp.int64(0) or heat <= wp.int64(0):
        return wp.int64(0)
    return wp.min(moved * heat / holds, moved)


@wp.func
def friction_law(crossing: wp.int64, friction: wp.float64, uniform: wp.float64) -> wp.int64:
    """How many of the quanta crossing a joint lose their direction there."""
    lost = wp.int64(wp.floor(wp.float64(crossing) * friction + uniform))
    return wp.max(wp.min(lost, crossing), wp.int64(0))


@wp.func
def heat_made(crossing: wp.int64, was_heat: wp.int64, lost: wp.int64) -> wp.int64:
    """How many of the quanta that lost their direction were not heat before."""
    if crossing <= wp.int64(0):
        return wp.int64(0)
    return lost * (crossing - was_heat) / crossing
