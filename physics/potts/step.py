"""Step: advance the universe by settings.step_time, in place: the energy fluid flows, the cells reshape.

Contract: `step(universe, settings)` changes the universe in place and returns None:
  0. on the first step only, the start's size correction (size_correction.py)
  1. cell_sums: each cell's energy, size, place, want, compression, clock
  2. pull: the potential of the A-energy (settings.potential_cycles V-cycles)
  3. cell_sums again, so each cell's mean potential is this step's
  4. energy_flow: one step of the energy fluid's equation (Gross-Pitaevskii, one
     complex number per box) on local clocks, every box at its own time level
  5. cell_sums, joint_swing (each joint's aim), then box_moves: one sweep of plate
     moves in which cell borders, edges and corners shift so sizes follow energy and
     the swing, the pull draws energy volumes towards each other through the joints
     and the joints push back, every energy volume staying wrapped in joints; the
     active region and the cell sums are refreshed
  6. joint_collapse: joints squeezed past their limit vanish and their energy volumes
     fall together; volume_split: volumes grown past the split size tear around a new
     joint; cell_sums after either
The pull's energy is the self-energy V n / 2 (fluid_energy.py), but the flow runs with V
held still, so it keeps V n, not V n / 2: the half of V's share the held field does not
follow, V n / 2 before the flow less after it (clocked, energy-volume boxes), is the
pull's work during the flow and goes to work_pull too (the re-solve then delivers it).
The work each non-flow phase does on the fluid's energy (fluid_energy.py, clocked: the
quantity the steps keep, in which friction's heat is booked, hop.py) is tallied, measured just before and after it: universe.work_pull (the pull's re-solve
and the cell sums after it), work_compression (the cell sums after the flow: the joints'
new compression changes their push and openness) and work_moves (the start's size
correction, joint swing, box moves, collapses and splits). Bookkeeping only: with heat
the fluid's energy balance closes as clocked fluid energy + heat - work, which only the
flow phase's splitting between time levels moves (stats.py "energy_balance").
universe.steps grows by 1 (substeps and box_updates by energy_flow); the last move
count is kept in universe.last_moves. Raises ValueError when a flow speed stops being
finite or a cell needs a time level over settings.max_level (the universe is then left
partly stepped).
"""
from __future__ import annotations

import warp as wp

from physics.potts.active_region import active_region
from physics.potts.box_moves import box_moves
from physics.potts.cell_sums import cell_sums
from physics.potts.energy_flow import energy_flow
from physics.potts.fluid_energy import fluid_energy
from physics.potts.joint_collapse import joint_collapse
from physics.potts.joint_swing import joint_swing
from physics.potts.pull import pull
from physics.potts.settings import Settings
from physics.potts.size_correction import size_correction
from physics.potts.universe import Universe
from physics.potts.volume_split import volume_split


def step(universe: Universe, settings: Settings) -> None:
    """One step: pull, the energy fluid on local clocks, one sweep of plate moves, collapses and splits."""
    cell_sums(universe, settings)
    before = fluid_energy(universe, settings, clocked=True)
    if not universe.corrected:
        size_correction(universe, settings)
        cell_sums(universe, settings)
        before = _tally(universe, settings, "work_moves", before)
    pull(universe, settings)
    cell_sums(universe, settings)
    _tally(universe, settings, "work_pull", before)
    held = _pull_energy(universe)
    energy_flow(universe, settings)
    universe.work_pull += held - _pull_energy(universe)
    before = fluid_energy(universe, settings, clocked=True)
    cell_sums(universe, settings)
    before = _tally(universe, settings, "work_compression", before)
    joint_swing(universe, settings)
    universe.last_moves = box_moves(universe, settings)
    active_region(universe)
    cell_sums(universe, settings)
    if joint_collapse(universe, settings) > 0:
        cell_sums(universe, settings)
    if volume_split(universe, settings) > 0:
        cell_sums(universe, settings)
    _tally(universe, settings, "work_moves", before)
    universe.steps += 1


def _tally(universe: Universe, settings: Settings, name: str, before: float) -> float:
    """Add the fluid energy's change since `before` to the named work tally; return the energy now."""
    now = fluid_energy(universe, settings, clocked=True)
    setattr(universe, name, getattr(universe, name) + now - before)
    return now


@wp.kernel
def _pull_sum(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32),
              potential: wp.array3d(dtype=wp.float32), psi: wp.array3d(dtype=wp.vec2d),
              clock: wp.array(dtype=wp.float32), low: wp.vec3i, total: wp.array(dtype=wp.float64)):
    ti, tj, tk = wp.tid()
    i = ti + low[0]
    j = tj + low[1]
    k = tk + low[2]
    cell = owner[i, j, k]
    if cell < 0:
        return
    if kind[cell] != 0:
        return
    held = psi[i, j, k]
    wp.atomic_add(total, 0, wp.float64(0.5 * clock[cell]) * wp.float64(potential[i, j, k]) * wp.dot(held, held))


def _pull_energy(universe: Universe) -> float:
    """The pull's clocked self-energy, sum over energy-volume boxes of clock V |psi|**2 / 2."""
    total = wp.zeros(1, dtype=wp.float64, device=universe.owner.device)
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    wp.launch(_pull_sum, dim=dims, device=universe.owner.device,
              inputs=[universe.owner, universe.kind, universe.potential, universe.psi, universe.cell_clock,
                      wp.vec3i(*universe.low), total])
    return float(total.numpy()[0])
