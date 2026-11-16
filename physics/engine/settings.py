"""Settings: the strength of every neighbour-only rule of the whole-quanta engine, all on by default.

Contract: frozen record of numbers read by step() and its kernels; `Settings()` is the
default run, every rule on, and an unknown name raises TypeError. Units: one Planck box
is length 1, one step lasts `step_time` of the counter, energy is counted in whole
quanta (int32 per box), hbar = box mass = 1, and the potentials below are energies per
quantum (they turn the phase, on_site.py). Defaults were set before any test ran and are
never changed to make a result pass.
  step_time        counter time per step
  light_speed      the pull's speed in boxes per unit time; light_speed * step_time must
                   stay below 1 / sqrt(3) (the leapfrog's limit on a cubic grid)
  pull_strength    G: in the static limit laplacian(pull) = 4 pi G (plus quanta per box);
                   sourced and felt by plus quanta only (bible: A pulls on A)
  volume_rest      quanta per plus box at which its fluid push is volume_stiffness
  volume_stiffness the plus fluid's own push per quantum at volume_rest, linear in the
                   packing (Gross-Pitaevskii: g n with g = volume_stiffness / volume_rest)
  joint_rest       quanta per minus box at which a joint is neither squeezed nor slack
  push             a joint's push per quantum at joint_rest
  push_steepness   the push rises as (packing / joint_rest) ** push_steepness, a hard wall
  seal_squeeze     squeeze (packing / joint_rest - 1) at which a crossing face closes
  collapse_squeeze squeeze past which a joint box gives way (minus quanta become plus);
                   larger than seal_squeeze, so a packed lump seals before it collapses
  friction         phase rub on a plus-minus face per unit time, times (1 + squeeze)
  spread_share     share of the quanta crossing a plus-minus face that spread from the
                   more packed box of the pair to the less packed (drift)
  clock_packing    quanta per box at which a local clock runs at half rate; 0 switches
                   local clocks off (side-by-side comparisons only)
  size_energy      the constant of bible rule 4, length = constant / energy: a cell of q
                   quanta wants to be size_energy / q boxes across, so more energy makes
                   it smaller (owner, 2026-11-16); default 1e7 = 10 boxes across at 1e6
                   quanta, the default start's cell (10**3 boxes of 1000), set before
                   any run so that start is at rest
  corner_mobility  how fast a lattice corner moves under the pressure of the cells round
                   it, in boxes per unit time per unit force (move_corners.py: each cell
                   presses by how much smaller than its energy wants it is, and the pull
                   draws energy volumes; no set size, no minimum)
  max_corner_step  numerics, not a rule: the most a corner moves in one step, in boxes,
                   so a box's covering cell changes only to a lattice neighbour
  rehomes_per_step how many of the 27 rehome phases (boxes 3 apart, drain.py and
                   hand_over.py) run each step; 27 checks every box every step
  max_turn         numerics, not a rule (owner 2026-11-16, decision E): the most the
                   phase difference across one same-sign face may turn in one fluid
                   substep, in radians; each step's fluid part is cut into as many equal
                   substeps per universe as its steepest face needs (substeps.py), so the
                   steep push is integrated stably; 0 runs one substep a step (the old,
                   coarse step, for side-by-side comparisons)
  max_substeps     the most substeps one step may take; a step that needs more takes
                   this many and is counted in state.capped
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Every rule's strength; the defaults are the run with all rules on."""

    step_time: float = 0.25
    light_speed: float = 1.0
    pull_strength: float = 1e-7
    volume_rest: float = 1000.0
    volume_stiffness: float = 0.5
    joint_rest: float = 1000.0
    push: float = 0.5
    push_steepness: float = 6.0
    seal_squeeze: float = 1.0
    collapse_squeeze: float = 2.0
    friction: float = 1.0
    spread_share: float = 0.01
    clock_packing: float = 10000.0
    size_energy: float = 1e7
    corner_mobility: float = 2e-5
    max_corner_step: float = 0.25
    rehomes_per_step: int = 27
    max_turn: float = 0.25
    max_substeps: int = 64

    def __post_init__(self) -> None:
        if self.light_speed * self.step_time >= 3.0 ** -0.5:
            raise ValueError("light_speed * step_time must stay below 1 / sqrt(3)")
        if not 0.0 < self.seal_squeeze < self.collapse_squeeze:
            raise ValueError("the seal must come before the collapse: 0 < seal_squeeze < collapse_squeeze")
        if not 0 <= self.rehomes_per_step <= 27:
            raise ValueError("rehomes_per_step runs between 0 and 27 phases")
        if not self.size_energy > 0.0:
            raise ValueError("size_energy must be > 0")
        if self.corner_mobility < 0.0 or not 0.0 <= self.max_corner_step < 1.0:
            raise ValueError("corner_mobility must be >= 0 and max_corner_step in 0 .. 1 box")
        if self.max_turn < 0.0 or not 1 <= self.max_substeps <= 1024:
            raise ValueError("max_turn must be >= 0 and max_substeps between 1 and 1024")
