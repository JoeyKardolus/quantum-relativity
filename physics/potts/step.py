"""Step: advance the universe by settings.step_time, in place: the fluid flows, the cells resize.

Contract: `step(universe, settings)` changes the universe in place and returns None:
  1. cell_sums: each cell's energy, size, want, compression, clock; the A source
  2. pull: the potential of the A-energy (settings.potential_cycles V-cycles)
  3. cell_sums again, so each cell's mean potential is this step's
  4. fluid substeps until step_time has passed on the counter, each with
     dt = min(courant / fastest_flow, time left): kick (pull on A, pressure inside a
     cell), then flow (exact donor-cell transport; friction and the seal only where
     energy crosses into or out of a joint); each box runs for its cell's clock * dt
  5. cell_sums, then box_moves: one sweep of eight parity phases in which cells claim
     or give up border boxes so their sizes follow their energy, the pull draws energy
     volumes towards each other through the joints and the joints push back
  6. cell_sums, then joint_collapse: joints squeezed past their limit vanish and their
     energy volumes fall together; cell_sums once more after a collapse
universe.steps grows by 1, substeps by the substeps taken; the last move count is
kept in universe.last_moves. Raises ValueError when a flow speed stops being finite
or a step would need more than settings.max_substeps (the universe is then left
partly stepped).
"""
from __future__ import annotations

from physics.potts.box_moves import box_moves
from physics.potts.cell_sums import cell_sums
from physics.potts.fastest_flow import fastest_flow
from physics.potts.flow import flow
from physics.potts.joint_collapse import joint_collapse
from physics.potts.kick import kick
from physics.potts.pull import pull
from physics.potts.settings import Settings
from physics.potts.universe import Universe


def step(universe: Universe, settings: Settings) -> None:
    """One step: pull, fluid substeps, one sweep of box moves, joint collapses."""
    cell_sums(universe, settings)
    pull(universe, settings)
    cell_sums(universe, settings)
    remaining = settings.step_time
    taken = 0
    while remaining > 1e-9 * settings.step_time:
        if taken >= settings.max_substeps:
            raise ValueError(f"step {universe.steps + 1}: more than {int(settings.max_substeps)} substeps needed")
        dt = min(remaining, settings.courant / fastest_flow(universe, settings))
        kick(universe, settings, dt)
        flow(universe, settings, dt)
        remaining -= dt
        taken += 1
    cell_sums(universe, settings)
    universe.last_moves = box_moves(universe, settings)
    cell_sums(universe, settings)
    if joint_collapse(universe, settings) > 0:
        cell_sums(universe, settings)
    universe.steps += 1
    universe.substeps += taken
