"""Step: advance the universe by settings.step_time, in place: the fluid flows, the cells reshape.

Contract: `step(universe, settings)` changes the universe in place and returns None:
  0. on the first step only, the start's size correction (size_correction.py)
  1. cell_sums: each cell's energy, size, place, want, compression, clock
  2. pull: the potential of the A-energy (settings.potential_cycles V-cycles)
  3. cell_sums again, so each cell's mean potential is this step's
  4. seam_flags marks the joints' seam boxes; local clocks (time_levels.py): each cell
     gets its own time level m, the finest
     M; then 2**M finest substeps, at each of which only the due boxes kick (pull on
     A, pressure; flow stopped at closed faces) and flow (exact donor-cell transport;
     friction and the seal only through a joint's seams), each box for its own dt =
     step_time / 2**m times its cell's clock: a packed, slow cell does few long
     updates, a fast one many short ones, and no cell waits on the fastest
  5. cell_sums, joint_swing (each joint's aim), then box_moves: one sweep of plate
     moves in which cell borders, edges and corners shift so sizes follow energy and
     the swing, the pull draws energy volumes towards each other through the joints,
     the joints push back and the links resist stretching; the active region and the
     cell sums are refreshed
  6. joint_collapse: joints squeezed past their limit vanish and their energy volumes
     fall together; volume_split: volumes grown past the split size tear around a new
     joint; cell_sums after either
universe.steps grows by 1, substeps by 2**M, box_updates by the box updates made; the
last move count is kept in universe.last_moves. Raises ValueError when a flow speed
stops being finite or a cell needs a time level over settings.max_level (the universe
is then left partly stepped).
"""
from __future__ import annotations

from physics.potts.active_region import active_region
from physics.potts.box_moves import box_moves
from physics.potts.cell_sums import cell_sums
from physics.potts.flow import flow
from physics.potts.joint_collapse import joint_collapse
from physics.potts.joint_swing import joint_swing
from physics.potts.kick import kick
from physics.potts.pull import pull
from physics.potts.seam_flags import seam_flags
from physics.potts.settings import Settings
from physics.potts.size_correction import size_correction
from physics.potts.time_levels import time_levels
from physics.potts.universe import Universe
from physics.potts.volume_split import volume_split


def step(universe: Universe, settings: Settings) -> None:
    """One step: pull, fluid substeps on local clocks, one sweep of plate moves, collapses and splits."""
    if not universe.corrected:
        size_correction(universe, settings)
    cell_sums(universe, settings)
    pull(universe, settings)
    cell_sums(universe, settings)
    finest = time_levels(universe, settings)
    seam_flags(universe)
    for substep in range(2 ** finest):
        threshold = 0 if substep == 0 else finest - _trailing_zeros(substep)
        kick(universe, settings, threshold)
        flow(universe, settings, threshold)
    counts = universe.scratch["kick_counts"]
    universe.box_updates += int(sum(int(counts[level]) * 2 ** level for level in range(finest + 1)))
    cell_sums(universe, settings)
    joint_swing(universe, settings)
    universe.last_moves = box_moves(universe, settings)
    active_region(universe)
    cell_sums(universe, settings)
    if joint_collapse(universe, settings) > 0:
        cell_sums(universe, settings)
    if volume_split(universe, settings) > 0:
        cell_sums(universe, settings)
    universe.steps += 1
    universe.substeps += 2 ** finest


def _trailing_zeros(number: int) -> int:
    return (number & -number).bit_length() - 1
