"""Pull wave law: the pull spreads from cell to cell as a wave, weaker with distance.

Contract: two Warp funcs, all float64.
  wave_weight(light_speed, step_time, length, neighbour_length)
      how much one neighbour counts: `distance = (length + neighbour_length) / 2` (a cell
      has no geometry, so every neighbour is half the one length plus half the other
      away, whatever its direction) and `weight = min(6 (light_speed step_time /
      distance) ** 2, 1)`. The 6 is not six neighbours: for N neighbours all at one
      distance, spread evenly round, `sum(neighbour - own) = N distance ** 2 / 6` times
      the second slope, so the mean over N times `6 / distance ** 2` is the right second
      slope for any N. The cap at 1 keeps the wave from going further than one
      neighbour a step. 0 when the distance is endless.
  wave_step(pull, pull_old, mean_weighted_difference, light_speed, step_time, source)
      the pull one step on: `2 pull - pull_old + mean_weighted_difference -
      (light_speed step_time) ** 2 source`, where `mean_weighted_difference` is the mean
      over the cell's neighbour entries of `weight * (neighbour's pull - own pull)` and
      `source` is pull_source.py. A source makes the pull deeper (more negative). No
      clock rate is in it: the pull is not something inside a cell, it spreads between
      cells, and a slow clock must not make a packed cell pull less (axiom 5: more
      packed pulls harder).
  wave_rest(mean_weighted_difference, light_speed, step_time, source)
      what `wave_step` adds to a pull that stood still: 0 everywhere, up to one level
      shared by all cells, is the pull at rest for the energy as it lies.
The bulk of a change travels near `light_speed`, measured about 1.6 times too fast (half of it is 4 neighbours on at step 2,143, light at 2,480), and a trace runs one neighbour a step. The step is taken from the nearest
neighbours, so a trace of a change (about `weight`, 1.5e-5 at the start, of it per
neighbour) is one neighbour further each step, ahead of light: see "Not done" in the
axiom's CLAUDE.md.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def wave_weight(light_speed: wp.float64, step_time: wp.float64, length: wp.float64,
                neighbour_length: wp.float64) -> wp.float64:
    """How much one neighbour counts in the wave step, at most 1."""
    distance = (length + neighbour_length) / wp.float64(2.0)
    if not (distance > wp.float64(0.0)) or wp.isinf(distance):
        return wp.float64(0.0)
    reached = light_speed * step_time / distance
    return wp.min(wp.float64(6.0) * reached * reached, wp.float64(1.0))


@wp.func
def wave_rest(mean_weighted_difference: wp.float64, light_speed: wp.float64, step_time: wp.float64,
              source: wp.float64) -> wp.float64:
    """What one step adds to a pull that stood still."""
    reach = light_speed * step_time
    return mean_weighted_difference - reach * reach * source


@wp.func
def wave_step(pull: wp.float64, pull_old: wp.float64, mean_weighted_difference: wp.float64,
              light_speed: wp.float64, step_time: wp.float64, source: wp.float64) -> wp.float64:
    """The pull of a cell one step on."""
    return wp.float64(2.0) * pull - pull_old + wave_rest(mean_weighted_difference, light_speed, step_time, source)
