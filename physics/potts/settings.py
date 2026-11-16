"""Settings: the strength of each rule in the cell engine; every rule is on in a run.

Contract: frozen record of floats, read by step() and stats(); `Settings()` and
`Settings.all_on()` are the defaults, and an unknown name raises TypeError. Units: one Planck box is
length 1 and volume 1, one step lasts `step_time`, and the start's mean energy per
occupied box is 1 (new_universe.py; energy volumes and joints alike).
  attraction            G in the pull: laplacian(potential) = 4 pi G (A-energy per box);
                        sourced by A only, felt by A only (bible: A pulls on A)
  size_power            how a cell's wanted box count follows its energy, one law for
                        both kinds: want = labels * k**3 * (labels * cell_rest / E)**size_power
                        (rule 4, length = constant / energy, so volume ~ energy**-3;
                        rule 8, a joint's size follows its own energy like a volume's)
  stiffness             size energy of a cell off its aim: stiffness * E * ln(n / aim)**2
                        (aim = want for an energy volume, the swing's aim for a joint)
  surface               border energy per box face a cell shares with another cell or
                        empty space, in units of its energy per box (box_moves.py; the
                        cellular Potts method's border term): keeps cells compact
  push                  a joint's push, a pressure P = push * c**push_power per box,
                        c = want / n, the joint's compression (B squeezed when c > 1);
                        in the energy fluid the same push stiffens a joint's fluid: its
                        nonlinear term is g * (1 + push * c**push_power) (on_site.py)
  push_power            how steeply the push rises as a joint is squeezed (> 0)
  seal_compression      c at or past which nothing crosses the joint's border (the seal)
  collapse_compression  c at or past which the joint collapses: it vanishes and its A
                        neighbours fall together into one energy volume
  friction              across an A-B face the flow crossing it (the phase difference as far
                        as the flow behind it on both sides carries it) keeps
                        1 / (1 + friction * c * dt) of itself per hop of length dt, the
                        lost flow energy booked as heat (hop.py); nowhere else is there
                        friction
  sound                 the energy fluid's sound speed at the start's mean density, both
                        kinds: its nonlinear term g = sound**2 / rest (on_site.py);
                        also sets a joint's swing, omega = pi * sound / n**(1/3)
                        (joint_swing.py)
  clock_scale           s in a cell's clock rate (s + 1) / (s + q), q its concentration
  step_time             counter time per step
  courant               each cell's own substep keeps (|u| + local sound) * clock * dt
                        under this (time_levels.py; u the flow from the phase)
  potential_cycles      multigrid V-cycles per step, warm-started
  max_level             a cell needing a substep finer than step_time / 2**max_level
                        raises ValueError (runaway guard)
  move_share            a cell may gain, and lose, at most this share of its boxes per
                        sweep of box moves (one sweep per step), at least one plate (4)
  split_boxes           an energy volume holding more than split_boxes * k**3 boxes tears
                        in two, the tear a new joint (rule 8, volume split; size, not
                        density, is the trigger)
  proposals             plate moves a site tries per sweep (box_moves.py)
  lump_factor           an A cell is a lump when its energy per box is this many times
                        the start's mean
Defaults were chosen once as units or carried over, not tuned: attraction 0.005, push 0.1,
push_power 3, seal (1/0.6)**3, collapse (1/0.5)**3, friction 1, lump_factor 2 from the
earlier fluid engine; size_power 3 from rule 4; stiffness 1, sound 1,
clock_scale 1, step_time 1, courant 0.4, potential_cycles 2 as units; move_share 1/8 (an
eighth of a cell per step, as before); split_boxes 8 (twice a start cell's width);
proposals 4; surface 0.1 (a tenth of a box's energy per border face, chosen once as a
unit-scale value, not fitted); max_level 20 (a million substeps, the old runaway guard).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    """Pull, size, push, seal, collapse, friction, fluid sound, clocks, time levels, moves, splits."""

    attraction: float = 0.005
    size_power: float = 3.0
    stiffness: float = 1.0
    surface: float = 0.1
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
    max_level: float = 20.0
    move_share: float = 1.0 / 8.0
    split_boxes: float = 8.0
    proposals: float = 4.0
    lump_factor: float = 2.0

    @classmethod
    def all_on(cls, **overrides: float) -> "Settings":
        """Defaults with every rule on, plus the given overrides."""
        return replace(cls(), **overrides)
