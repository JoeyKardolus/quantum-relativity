"""Settings: the strength of every rule of the whole-quanta engine, all on by default.

Contract: frozen record of numbers read by step() and its kernels; `Settings()` is the
default run, every rule on, and an unknown name raises TypeError. Units: one Planck box
is length 1, one step lasts `step_time` of the counter, energy is counted in whole
quanta (int32 per box), hbar = box mass = 1, and pull and push are energies per quantum.
Defaults were set before any test ran and are never changed to make a result pass.
There is no size at rest and no drift setting: a cell's wanted size comes from its own
energy (length = constant / energy, the constant fixed once by the start's correction,
state.size_energy), an energy volume only pulls, a joint only pushes, empty space does
nothing.
  step_time        counter time per step
  light_speed      the pull's speed in boxes per unit time; light_speed * step_time must
                   stay below 1 / sqrt(3) (the leapfrog's limit on a cubic grid)
  pull_strength    G: in the static limit laplacian(pull) = 4 pi G (plus quanta per box);
                   sourced and felt by plus quanta only (bible: A pulls on A)
  push             a joint's push per quantum when it holds the size its energy wants
  push_steepness   the push rises as (wanted size / size) ** push_steepness, a hard wall
  seal_squeeze     squeeze (wanted length / length - 1) at which a joint lets nothing cross
  collapse_packing minus quanta per box past which a joint collapses and its two volumes
                   become one (owner, 2026-11-16: a joint collapses only when its energy
                   concentration gets too high, never because it is low on energy or
                   smaller than its energy wants); default the start's 1000 quanta per
                   box pressed into a 27th of the room (a side 3 times shorter), the
                   limit the earlier collapse squeeze of 2 set for a joint at the start's
                   packing
  crossing_rate    energy crosses a joint by diffusion bent by the pull (owner,
                   2026-11-16; cross.py): per unit of the joint's local time the quanta
                   moved from one volume to the other are crossing_rate * openness *
                   (B(x) Q0 - B(-x) Q1), plain difference of their quanta when the pulls
                   are equal; the drift's strength comes from the fluid's own jiggle,
                   no setting of its own; default the fluid's own diffusion constant hbar / 2m = 1/2 box^2 per
                   time over the start's cell spacing of 11 boxes squared (0.5 / 121)
  joint_inertia    a joint's size accelerates by the push differences with the joints
                   it shares a volume with, over this inertia (natural units: 1)
  clock_packing    quanta per box at which a cell's local clock runs at half rate; 0
                   switches local clocks off (side-by-side comparisons only)
  max_turn         numerics, not a rule: the most the phase difference across one face
                   inside a cell may turn in one fluid substep, in radians; 0 runs one
                   substep a step
  max_substeps     the most fluid substeps one step may take; a step that needs more
                   takes this many and is counted in state.capped
  split_across     the size across, in boxes (cube root of its box count), at which an
                   energy volume tears in two and the tear becomes a new joint (bible
                   rule 8: the trigger is size); 0 switches splits off (comparisons only)
  least_cell_boxes the fewest Planck boxes an energy volume may start with or keep after a tear
                   (owner, 2026-11-16: a cell needs enough boxes to hold a proper whirl);
                   measured, not fitted: the smallest cube in which a placed whirl (core
                   tanh(r)^2, uneven energy, seeds 1 and 2) keeps its winding round every
                   loop for 1000 steps with its core under half the cell's mean packing
                   is 6 boxes across; at 5 across the whirl leaves the middle by step
                   850-1000, at 4 its core never opens (up to 0.69 of the mean)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Every rule's strength; the defaults are the run with all rules on."""

    step_time: float = 0.25
    light_speed: float = 1.0
    pull_strength: float = 1e-7
    push: float = 0.5
    push_steepness: float = 6.0
    seal_squeeze: float = 1.0
    collapse_packing: float = 27000.0
    crossing_rate: float = 0.5 / 121.0
    joint_inertia: float = 1.0
    clock_packing: float = 10000.0
    max_turn: float = 0.25
    max_substeps: int = 64
    split_across: float = 20.0
    least_cell_boxes: int = 216

    def __post_init__(self) -> None:
        if self.light_speed * self.step_time >= 3.0 ** -0.5:
            raise ValueError("light_speed * step_time must stay below 1 / sqrt(3)")
        if not (self.seal_squeeze > 0.0 and self.collapse_packing > 0.0):
            raise ValueError("seal_squeeze and collapse_packing must be > 0")
        if self.crossing_rate < 0.0 or self.joint_inertia <= 0.0:
            raise ValueError("crossing_rate must be >= 0 and joint_inertia > 0")
        if self.least_cell_boxes < 1:
            raise ValueError("least_cell_boxes must be at least 1")
        if self.split_across < 0.0:
            raise ValueError("split_across must be >= 0 (0: no splits)")
        if self.max_turn < 0.0 or not 1 <= self.max_substeps <= 1024:
            raise ValueError("max_turn must be >= 0 and max_substeps between 1 and 1024")
