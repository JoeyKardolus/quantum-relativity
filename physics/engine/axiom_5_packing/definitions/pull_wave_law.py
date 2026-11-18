"""Pull wave law: the pull spreads from cell to cell as a wave, weaker with distance.

Contract: three Warp funcs, all float64. A cell has no geometry, so every neighbour is
half the one length plus half the other away, whatever its direction; a cell's `reach`
is the mean of those distances over the neighbours it hears (heard_pull.py), the room
it stands in.
  wave_weight(light_speed, step_time, reach)
      how far the wave gets across the cell's reach in one step: `min((light_speed
      step_time / reach) ** 2, 1)`. Every neighbour counts as `reach` away, so along
      any one line of neighbours the wave then runs at `light_speed` at most (along a
      row of the start lattice, where 18 of a cell's 26 neighbours lie ahead or behind,
      at 0.59 of it), never above. The cap at 1 lets a cell too small for the step take
      the wave on slower, never further than one neighbour a step. 0 for an endless
      reach.
  wave_rest(heard, reach, source)
      what stands against the source: `heard - reach ** 2 * source / 6`, `heard` the
      mean over the cell's neighbours of `(distance / reach) * (neighbour's pull - own
      pull)` and `source` pull_source.py (the 6 is not six neighbours: for N neighbours
      all at one distance, spread evenly round, `sum(neighbour - own) = N distance ** 2
      / 6` times the second slope). 0 everywhere, up to what one level shared by
      all cells leaves, is the pull at rest for the energy as it lies.
  wave_step(pull, pull_old, weight, heard, reach, source)
      the pull one step on: `2 pull - pull_old + weight * wave_rest`. A source makes
      the pull deeper (more negative). No clock rate is in it: the pull is not
      something inside a cell, it spreads between cells, and a slow clock must not make
      a packed cell pull less (axiom 5).
Why the neighbours count by their distance: at rest `sum over neighbours of distance *
(neighbour's pull - own pull)` equals the cell's quanta times a constant. The same
number stands on both sides of every relation, so what leaves a cell is its energy
(far away the pull goes as the energy over the distance), two cells pull each other
equally hard, and a cell packed into less room sits deeper: more packed pulls harder.
(Until 2026-11-18 a neighbour counted by one over its distance squared; a fuller, so
smaller, volume then sat shallower than its neighbours.)
Measured on the balanced start (16 a side): neither half nor a hundredth of a change
is 1 to 4 neighbours away before light is.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})

_SIX = wp.constant(wp.float64(6.0))


@wp.func
def wave_weight(light_speed: wp.float64, step_time: wp.float64, reach: wp.float64) -> wp.float64:
    """How far the wave gets across a cell's reach in one step, at most 1."""
    if not (reach > wp.float64(0.0)) or wp.isinf(reach):
        return wp.float64(0.0)
    reached = light_speed * step_time / reach
    return wp.min(reached * reached, wp.float64(1.0))


@wp.func
def wave_rest(heard: wp.float64, reach: wp.float64, source: wp.float64) -> wp.float64:
    """What a cell hears of its neighbours less what its own source asks for; 0 at rest."""
    return heard - reach * reach * source / _SIX


@wp.func
def wave_step(pull: wp.float64, pull_old: wp.float64, weight: wp.float64, heard: wp.float64, reach: wp.float64,
              source: wp.float64) -> wp.float64:
    """The pull of a cell one step on."""
    return wp.float64(2.0) * pull - pull_old + weight * wave_rest(heard, reach, source)
