"""Step: one step of the whole-quanta engine, every rule neighbour-only.

Contract: `step(state, settings)` advances the state by one step_time of the counter:
  1. pull_wave     the pull moves one leapfrog step at light speed (pull_wave.py)
  2. fluid         Strang splitting: on_site for half a step, the hops along x, y, z
                   (lower-box parity 0 then 1) for half a step each, the same hops in
                   reverse order for the other half, on_site for half a step (on_site.py,
                   hop.py: whole quanta, exact amount, friction and drift spreading on
                   plus-minus faces, local clocks)
  3. claim         `claims_per_step` of the 64 claim phases, in an order shuffled afresh
                   every 64 / claims_per_step steps, one fixed sequence for every universe so a
                   batch steps each universe as it would alone (claim.py:
                   border moves, wrapping kept, space grows into empty boxes)
  4. collapse      joint boxes past the limit turn plus (collapse.py)
  5. tick_clocks   every box's clock advances by its own proper time (tick_clocks.py)
then counts the step and, when a detail ladder is attached (attach_ladder.py), writes the
changed blocks into it and switches every block's level (ladder_sync.py). The amount
ignoring sign is exact through every part. Returns None.
"""
from __future__ import annotations

import numpy as np

from physics.engine.claim import claim
from physics.engine.collapse import collapse
from physics.engine.hop import hop
from physics.engine.ladder_sync import ladder_sync
from physics.engine.on_site import on_site
from physics.engine.pull_wave import pull_wave
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.tick_clocks import tick_clocks

_ORDER = [(axis, parity) for axis in range(3) for parity in range(2)]
_CLAIM_ORDER_SEED = 64


def step(state: State, settings: Settings) -> None:
    """One step: pull wave, fluid with whole quanta, border claims, collapse, local clocks."""
    pull_wave(state, settings)
    half = 0.5 * settings.step_time
    on_site(state, settings, half)
    for pass_index, (axis, parity) in enumerate(_ORDER + _ORDER[::-1]):
        hop(state, settings, axis, parity, half, pass_index)
    on_site(state, settings, half)
    for phase_index in _claim_phases(state, settings):
        claim(state, settings, phase_index)
    collapse(state, settings)
    tick_clocks(state, settings)
    state.step += 1
    ladder_sync(state)


def _claim_phases(state: State, settings: Settings) -> list:
    """The claim phases this step runs: a slice of a fresh shuffle of the 64, one shuffle per sweep."""
    per_step = settings.claims_per_step
    if per_step == 0:
        return []
    steps_per_sweep = -(-64 // per_step)
    sweep, place = divmod(state.step, steps_per_sweep)
    order = np.random.default_rng((_CLAIM_ORDER_SEED, sweep)).permutation(64)
    return [int(index) for index in order[place * per_step:(place + 1) * per_step]]
