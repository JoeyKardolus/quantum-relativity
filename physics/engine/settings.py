"""Settings: the strength of every rule of the cell-grid engine, all on by default.

Contract: frozen record of numbers read by step() and its kernels; `Settings()` is the
default run, every rule on, and an unknown name raises TypeError. Units: a cell's length
is its size (state.py: constant / quanta; an energy volume starts about 10 long, a joint
about 1), one step lasts `step_time` of the counter, energy is counted in whole quanta,
and inside a cell's compute boxes hbar = box mass = 1. Every default was set before any
run and is never changed to make a result pass; the fit may search them.
  step_time       counter time per step
  light_speed     the pull's speed, in length per unit of local time
  pull_strength   G: the pull's source is 4 pi G (plus quanta / length ** 3) and two
                  volumes pull on each other across a joint with G * quanta * quanta /
                  distance ** 2 (squeeze.py); sourced and felt by plus quanta only
  push_strength   P: a joint pushes back with P * (its quanta / its length) ** 2, which
                  is P * quanta ** 4 / constant ** 2: steep in its energy
  squeeze_rate    a squeezed joint (pull across it above its push) makes minus quanta
                  out of itself at squeeze_rate * quanta * (squeeze - 1) per unit of its
                  local time, never past the energy at which its push balances the pull
  seal_thin       how many times thinner than the start's mean joint a joint is when it
                  lets nothing cross; it lets everything cross at the start's thickness
                  or fatter, and falls off in a straight line between
  collapse_thin   how many times thinner than the start's mean joint a squeezed joint
                  is when it collapses (the earlier engine's limit: a side 3 times shorter)
  crossing_rate   plus quanta crossing a joint per unit of its local time are
                  crossing_rate * openness * (B(x) Qa - B(-x) Qb) (cross.py): plain
                  difference when the pulls are equal, bent toward the deeper pull
  tear_length     the length at which an energy volume, grown that big by losing
                  energy, tears in two (twice the start's mean volume); 0: no tears
                  (side-by-side comparisons only)
  clock_packing   packing (start length of its kind / length, cubed) at which a cell's
                  local clock runs at half rate; 0 switches local clocks off
                  (comparisons only)
  fluid_pressure  the fluid's own pressure inside a cell per quantum, against the
                  cell's mean per box (fluid_potential.py); a whirl's core is
                  1 / sqrt(2 fluid_pressure) boxes wide
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Every rule's strength; the defaults are the run with all rules on."""

    step_time: float = 0.25
    light_speed: float = 1.0
    pull_strength: float = 1e-7
    push_strength: float = 2e-6
    squeeze_rate: float = 1.0
    seal_thin: float = 2.0
    collapse_thin: float = 3.0
    crossing_rate: float = 0.5 / 121.0
    tear_length: float = 20.0
    clock_packing: float = 10.0
    fluid_pressure: float = 0.5

    def __post_init__(self) -> None:
        if not (self.step_time > 0.0 and self.light_speed > 0.0):
            raise ValueError("step_time and light_speed must be > 0")
        if self.pull_strength < 0.0 or self.push_strength <= 0.0:
            raise ValueError("pull_strength must be >= 0 and push_strength > 0")
        if self.squeeze_rate < 0.0 or self.crossing_rate < 0.0:
            raise ValueError("squeeze_rate and crossing_rate must be >= 0")
        if not 1.0 < self.seal_thin or not 1.0 < self.collapse_thin:
            raise ValueError("seal_thin and collapse_thin must be above 1")
        if self.tear_length < 0.0:
            raise ValueError("tear_length must be >= 0 (0: no tears)")
        if self.fluid_pressure < 0.0:
            raise ValueError("fluid_pressure must be >= 0")
