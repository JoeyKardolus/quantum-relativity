"""Settings: the strength of each rule in the cell engine; every rule is on in a run.

Contract: frozen record of floats, read by step() and stats(); `Settings()` and
`Settings.all_on()` are the defaults, and an unknown name raises TypeError. Units: one Planck box is
length 1 and volume 1, one step lasts `step_time`, and the start's mean A-energy per
A box is 1 (new_universe.py).
  attraction            G in the pull: laplacian(potential) = 4 pi G (A-energy per box);
                        sourced by A only, felt by A only (bible: A pulls on A)
  size_power            how a cell's wanted box count follows its energy:
                        want = labels * k**3 * (labels * mean_cell_energy / energy)**size_power
                        (rule 4, length = constant / energy, so volume ~ energy**-3)
  stiffness             size energy of a cell off its want: stiffness * E * (n / want - 1)**2
  push                  a joint's push, a pressure P = push * c**push_power per box,
                        c = want / n, the joint's compression (B squeezed when c > 1)
  push_power            how steeply the push rises as a joint is squeezed (> 0)
  seal_compression      c at or past which nothing crosses the joint's border (the seal)
  collapse_compression  c at or past which the joint collapses: it vanishes and its A
                        neighbours fall together into one energy volume
  friction              crossing an A-B border keeps 1 / (1 + friction * c) of the
                        crossing energy's momentum; nowhere else is there friction
  sound                 the fluid's sound speed in energy volumes: pressure = sound**2 *
                        rest_a * (energy per box / its kind's rest per box), kick.py
  clock_scale           s in a cell's clock rate (s + 1) / (s + q), q its concentration
  step_time             counter time per step
  courant               the substep keeps (|u| + sound) * clock * dt under this
  potential_cycles      multigrid V-cycles per step, warm-started
  max_substeps          a step needing more raises ValueError
  move_share            a cell may gain, and lose, at most this share of its boxes per
                        move phase (eight phases per step), at least 1
  lump_factor           an A cell is a lump when its energy per box is this many times
                        the start's mean
Defaults were chosen once as units or carried over from the fluid engine, not tuned:
attraction 0.005, push 0.1, push_power 3, seal (1/0.6)**3, collapse (1/0.5)**3,
friction 1, lump_factor 2 from the earlier fluid engine; size_power 3 from rule 4; stiffness 1,
sound 1, clock_scale 1, step_time 1, courant 0.4, potential_cycles 2 as units;
move_share 1/64 (a cell changes by at most an eighth of its boxes per step); max_substeps 1e6, a
runaway guard only: steps of up to 7,761 substeps were measured, so 2000 stopped default runs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    """Pull, size, push, seal, collapse, friction, fluid sound, clocks, time stepping, move limits."""

    attraction: float = 0.005
    size_power: float = 3.0
    stiffness: float = 1.0
    push: float = 0.1
    push_power: float = 3.0
    seal_compression: float = (1.0 / 0.6) ** 3
    collapse_compression: float = (1.0 / 0.5) ** 3
    friction: float = 1.0
    sound: float = 1.0
    clock_scale: float = 1.0
    step_time: float = 1.0
    courant: float = 0.4
    potential_cycles: float = 2.0
    max_substeps: float = 1e6
    move_share: float = 1.0 / 64.0
    lump_factor: float = 2.0

    @classmethod
    def all_on(cls, **overrides: float) -> "Settings":
        """Defaults with every rule on, plus the given overrides."""
        return replace(cls(), **overrides)

