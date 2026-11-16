"""Step: one step of the whole-quanta engine, every rule neighbour-only.

Contract: `step(state, settings)` advances the state by one step_time of the counter:
  1. pull_wave     the pull moves one leapfrog step at light speed (pull_wave.py)
  2. move_corners  every cell's quanta are counted and the forced cell lattice's
                   corners move by the cells' pressures and the pull, so each cell's
                   size follows its energy (move_corners.py)
  3. fluid         cut into equal substeps per universe, as many as its steepest
                   same-sign face needs (substeps.py; numerics, owner decision E), each
                   a Strang splitting: on_site for half a step, the hops along x, y, z
                   (lower-box parity 0 then 1) for half a step each, the leaks across
                   joints along x, y, z (group 0 then 1) for half a step each and
                   again in reverse order, the hops in reverse order for the other half,
                   on_site for half a step (on_site.py; hop.py: whole quanta inside one
                   sign, frictionless; leak.py: plus quanta from energy volume box to
                   energy volume box across one joint in one transfer, friction and
                   drift spreading; local clocks)
  4. rehome        box_targets finds the cell covering each box, then `rehomes_per_step`
                   of the 27 rehome phases, in an order shuffled afresh every 27 /
                   rehomes_per_step steps, one fixed sequence for every universe so a
                   batch steps each universe as it would alone, each a drain (a box
                   leaving its cell empties into its own cell, drain.py) and a hand
                   over (the emptied box joins the covering cell, hand_over.py): space
                   grows where outer joints take empty boxes
  5. collapse      joint boxes past the limit turn plus (collapse.py)
  5b. volume_split the largest energy volume grown past split_across in each universe
                   tears in two, new lattice planes, the tear a new joint
                   (volume_split.py)
  6. tick_clocks   every box's clock advances by its own proper time (tick_clocks.py)
then counts the step and, when a detail ladder is attached (attach_ladder.py), writes the
changed blocks into it and switches every block's level (ladder_sync.py). The amount
ignoring sign is exact through every part, and quanta keep their sign except in collapse.
`marks` is zeroed first; owner changes mark their boxes (2) and collapse its flipped
boxes (1), a split its relabelled and torn boxes (8), so after a step `marks` says
which boxes changed cell or sign and by which rule.
Returns None.
"""
from __future__ import annotations

import numpy as np

from physics.engine.box_targets import box_targets
from physics.engine.collapse import collapse
from physics.engine.drain import drain
from physics.engine.hand_over import hand_over
from physics.engine.hop import hop
from physics.engine.ladder_sync import ladder_sync
from physics.engine.leak import leak
from physics.engine.move_corners import move_corners
from physics.engine.on_site import on_site
from physics.engine.pull_wave import pull_wave
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.substeps import substeps
from physics.engine.tick_clocks import tick_clocks
from physics.engine.volume_split import volume_split

_ORDER = [(axis, parity) for axis in range(3) for parity in range(2)]
_REHOME_ORDER_SEED = 27
_PHASES = 27


def step(state: State, settings: Settings) -> None:
    """One step: pull wave, moving cell corners, fluid in substeps, boxes rehomed, collapse, local clocks."""
    state.marks.zero_()
    pull_wave(state, settings)
    move_corners(state, settings)
    half = 0.5 * settings.step_time
    for sub_index in range(int(substeps(state, settings).max())):
        _fluid(state, settings, half, sub_index)
    box_targets(state)
    for phase_index in _rehome_phases(state, settings):
        drain(state, phase_index)
        hand_over(state, phase_index)
    collapse(state, settings)
    volume_split(state, settings)
    tick_clocks(state, settings)
    state.step += 1
    ladder_sync(state)


def _fluid(state: State, settings: Settings, half: float, sub_index: int) -> None:
    """One fluid substep, Strang split; each kernel divides `half` by its universe's substep count."""
    on_site(state, settings, half, sub_index)
    for pass_index, (axis, parity) in enumerate(_ORDER):
        hop(state, settings, axis, parity, half, pass_index, sub_index)
    for pass_index, (axis, group) in enumerate(_ORDER + _ORDER[::-1]):
        leak(state, settings, axis, group, half, 12 + pass_index, sub_index)
    for pass_index, (axis, parity) in enumerate(_ORDER[::-1]):
        hop(state, settings, axis, parity, half, 6 + pass_index, sub_index)
    on_site(state, settings, half, sub_index)


def _rehome_phases(state: State, settings: Settings) -> list:
    """The rehome phases this step runs: a slice of a fresh shuffle of the 27, one shuffle per sweep."""
    per_step = settings.rehomes_per_step
    if per_step == 0 or not state.has_lattice:
        return []
    steps_per_sweep = -(-_PHASES // per_step)
    sweep, place = divmod(state.step, steps_per_sweep)
    order = np.random.default_rng((_REHOME_ORDER_SEED, sweep)).permutation(_PHASES)
    return [int(index) for index in order[place * per_step:(place + 1) * per_step]]
