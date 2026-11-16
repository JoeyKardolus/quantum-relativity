"""Step: one step of the whole-quanta engine, every rule neighbour-only.

Contract: `step(state, settings)` advances the state by one step_time of the counter:
  1. pull_wave     the pull moves one leapfrog step at light speed (pull_wave.py)
  2. fluid         cut into equal substeps per universe, as many as its steepest
                   same-sign face needs (substeps.py; numerics, owner decision E), each
                   a Strang splitting: on_site for half a step, the hops along x, y, z
                   (lower-box parity 0 then 1) for half a step each, the leaks across
                   joints along x, y, z (group 0 then 1) for half a step each and
                   again in reverse order, the hops in reverse order for the other half,
                   on_site for half a step (on_site.py; hop.py: whole quanta inside one
                   sign, frictionless; leak.py: plus quanta from energy volume box to
                   energy volume box across one joint in one transfer, friction and
                   drift spreading; local clocks)
  3. claim         `claims_per_step` of the 64 claim phases, in an order shuffled afresh
                   every 64 / claims_per_step steps, one fixed sequence for every universe so a
                   batch steps each universe as it would alone (claim.py:
                   border moves, wrapping kept, space grows into empty boxes)
  4. collapse      joint boxes past the limit turn plus (collapse.py)
  5. tick_clocks   every box's clock advances by its own proper time (tick_clocks.py)
then counts the step and, when a detail ladder is attached (attach_ladder.py), writes the
changed blocks into it and switches every block's level (ladder_sync.py). The amount
ignoring sign is exact through every part, and quanta keep their sign except in collapse.
`marks` is zeroed first; claims mark their sites (2) and collapse its flipped boxes (1),
so after a step `marks` says which boxes changed kind and by which rule.
Returns None.
"""
from __future__ import annotations

import numpy as np

from physics.engine.claim import claim
from physics.engine.collapse import collapse
from physics.engine.hop import hop
from physics.engine.ladder_sync import ladder_sync
from physics.engine.leak import leak
from physics.engine.on_site import on_site
from physics.engine.pull_wave import pull_wave
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.substeps import substeps
from physics.engine.tick_clocks import tick_clocks

_ORDER = [(axis, parity) for axis in range(3) for parity in range(2)]
_CLAIM_ORDER_SEED = 64


def step(state: State, settings: Settings) -> None:
    """One step: pull wave, fluid with whole quanta in substeps, border claims, collapse, local clocks."""
    state.marks.zero_()
    pull_wave(state, settings)
    half = 0.5 * settings.step_time
    for sub_index in range(int(substeps(state, settings).max())):
        _fluid(state, settings, half, sub_index)
    for phase_index in _claim_phases(state, settings):
        claim(state, settings, phase_index)
    collapse(state, settings)
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


def _claim_phases(state: State, settings: Settings) -> list:
    """The claim phases this step runs: a slice of a fresh shuffle of the 64, one shuffle per sweep."""
    per_step = settings.claims_per_step
    if per_step == 0:
        return []
    steps_per_sweep = -(-64 // per_step)
    sweep, place = divmod(state.step, steps_per_sweep)
    order = np.random.default_rng((_CLAIM_ORDER_SEED, sweep)).permutation(64)
    return [int(index) for index in order[place * per_step:(place + 1) * per_step]]
