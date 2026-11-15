"""Settings: the strength of each fluid rule; every rule is always on in a scene.

Contract: frozen record of floats, read by step() and stats(). Units: one box is
length 1, one step lasts `step_time`, and the start's mean A-energy per box in the
ball is 1 (new_grid.py).
  attraction            G in the pull: laplacian(potential) = 4 pi G (A-energy per box);
                        the pull acts on A-energy only (bible: A pulls on A)
  push                  B pressure strength: P = push * b_rest * c**push_power / push_power,
                        c = (B-energy per box) / b_rest, the B compression
  push_power            how steeply B pressure rises with compression (> 0)
  b_rest                B-energy per box at which c = 1; the start's mean (new_grid.JOINT_SHARE)
  seal_compression      c at or past which A-energy cannot pass the B in a box: A is locked
                        to B there and no A crosses a face between a sealed and an unsealed
                        box; inf disables the seal (side-by-side comparison runs only)
  collapse_compression  c past which B pressure stops rising (the collapse limit): the
                        runaway, a black hole's core
  friction              A-through-B drag rate at c = 1; the rate is friction * c, so more
                        compressed B rubs harder; the only friction there is
  step_time             time one step() advances
  courant               CFL number: each substep lasts courant / (fastest wave sum per box)
  potential_cycles      multigrid V-cycles per substep, warm-started from the last potential
  max_substeps          step() raises ValueError when one step needs more substeps
  occupied_floor        a box is occupied (stats, view) when its energy passes this
                        fraction of the start's mean A per box
  lump_factor           a box is in a lump when its A-energy is at least this times the
                        start's mean A per box (abab's LUMP_FACTOR)
`Settings.all_on(**overrides)` gives the defaults. Carried from the earlier ABAB engine (git history), not tuned:
attraction 0.005, push 0.1, push_power 3, b_rest 0.03 (the joint's 3% price), the seal
at abab's seal_point 0.4 and the collapse at its push_limit 0.5 as density ratios
(1/0.6)**3 and (1/0.5)**3, lump_factor 2. Chosen as units, not tuned: friction 1,
step_time 1, courant 0.4, potential_cycles 2, max_substeps 2000, occupied_floor 0.001.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    """Pull, push and its power, rest, seal, collapse, friction, time stepping, thresholds."""

    attraction: float
    push: float
    push_power: float
    b_rest: float
    seal_compression: float
    collapse_compression: float
    friction: float
    step_time: float
    courant: float
    potential_cycles: float
    max_substeps: float
    occupied_floor: float
    lump_factor: float

    @classmethod
    def all_on(cls, **overrides: float) -> "Settings":
        """Defaults with every rule on, plus the given overrides."""
        return replace(_ALL_ON, **overrides)


_ALL_ON = Settings(attraction=0.005, push=0.1, push_power=3.0, b_rest=0.03,
                   seal_compression=(1.0 / 0.6) ** 3, collapse_compression=(1.0 / 0.5) ** 3,
                   friction=1.0, step_time=1.0, courant=0.4, potential_cycles=2.0,
                   max_substeps=2000.0, occupied_floor=1e-3, lump_factor=2.0)
