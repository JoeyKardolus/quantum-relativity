"""Step: one step of the cell-grid engine, every rule on its cell's local clock.

Contract: `step(state, settings)` advances the state by settings.step_time of the
counter and returns None. In order:
  cell_laws   every cell's size, packing and clock rate from its quanta (the size law)
  pull_wave   the pull spreads one step, from cell to cell at light speed
  on_site, hop  the fluid inside every cell: half a phase turn, the six passes of pair
              rotations over the faces of its own block, half a phase turn (Strang)
  jiggle      each energy volume's own heat (after the fluid moved: a start at phase 0
              has none before)
  squeeze     volumes pull across their joints; squeezed joints make minus quanta
  cross       plus quanta cross from volume to volume through the joint between them
  pour        what each cell gave and took leaves and enters its own boxes; arrivals heat
  tick_clocks every cell's clock moves on at its own rate
  events      the collapses and tears found, the only changes of who neighbours whom
  cell_laws   again, so every size is the law of the cell's quanta when the step ends
  quantum     every tracked moving energy's own step (quantum.py, axiom 7)
state.gain, state.loss, state.crossed, state.squeeze and state.flags hold this step's
numbers afterwards. Nothing here reads a box of one cell from another cell.
"""
from __future__ import annotations

from physics.engine.cell_laws import cell_laws
from physics.engine.cross import cross
from physics.engine.events import events
from physics.engine.hop import hop
from physics.engine.jiggle import jiggle
from physics.engine.on_site import on_site
from physics.engine.pour import pour
from physics.engine.pull_wave import pull_wave
from physics.engine.quantum import quantum
from physics.engine.settings import Settings
from physics.engine.squeeze import squeeze
from physics.engine.state import State
from physics.engine.tick_clocks import tick_clocks


def step(state: State, settings: Settings) -> None:
    """Advance every cell by one step of the counter, each rule at the cell's own clock rate."""
    for name in ("gain", "loss", "crossed", "found"):
        getattr(state, name).zero_()
    cell_laws(state, settings)
    pull_wave(state, settings)
    on_site(state, settings, 0.5 * settings.step_time)
    for axis in range(3):
        for parity in range(2):
            hop(state, settings, axis, parity, settings.step_time, 2 * axis + parity)
    on_site(state, settings, 0.5 * settings.step_time)
    jiggle(state)
    squeeze(state, settings)
    cross(state, settings)
    pour(state)
    tick_clocks(state, settings)
    if state.found.numpy().any():
        events(state, settings)
    cell_laws(state, settings)
    quantum(state, settings)
    state.step += 1
