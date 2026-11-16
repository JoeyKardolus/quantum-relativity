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
  claims_per_step  how many of the 64 claim phases (blocks 4 apart) run each step
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
    claims_per_step: int = 1

    def __post_init__(self) -> None:
        if self.light_speed * self.step_time >= 3.0 ** -0.5:
            raise ValueError("light_speed * step_time must stay below 1 / sqrt(3)")
        if not 0.0 < self.seal_squeeze < self.collapse_squeeze:
            raise ValueError("the seal must come before the collapse: 0 < seal_squeeze < collapse_squeeze")
        if not 0 <= self.claims_per_step <= 64:
            raise ValueError("claims_per_step runs between 0 and 64 phases")
