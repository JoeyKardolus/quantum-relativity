"""Settings: the strength of each fluid rule; every rule is always on in a scene.

Contract: frozen record of floats, read by step() and stats(). Units: one box is
length 1, one step lasts `step_time`, and the start's mean A-energy per box in the
filled cube is 1 (new_grid.py).
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
  leak                  the share of its A-energy per unit of its own time a box gives
                        through fully open joints to its 6 neighbours, split by their pull
                        (leak.py, bible rule 6); joint openness (joint_openness.py) narrows
                        it with squeeze and closes it at seal_compression
  clock_scale           how strongly energy concentration slows a box's clock
                        (clock_rate.py): rate = (clock_scale + 1) / (clock_scale + q), q the
                        box's concentration (1 at the start's mean); the larger, the weaker
  step_time             time one step() advances
  courant               CFL number: each substep lasts courant / (fastest wave sum per box)
  potential_cycles      multigrid V-cycles per substep, warm-started from the last potential
  max_substeps          step() raises ValueError when one step needs more substeps
  cfl_floor             a kind (A or B) holding less energy in a box than this fraction of
                        the start's mean A per box does not set the substep (fastest_wave.py);
                        its outflow is capped at its content instead (transport.py), so
                        near-empty vapour falling fast into a deep core cannot shrink every
                        box's time step
  occupied_floor        a box is occupied (stats, view) when its energy passes this
                        fraction of the start's mean A per box
  lump_factor           a box is in a lump when its A-energy is at least this times the
                        start's mean A per box (abab's LUMP_FACTOR)
`Settings.all_on(**overrides)` gives the defaults. Carried from the earlier ABAB engine (git history), not tuned:
attraction 0.005, push 0.1, push_power 3, b_rest 0.03 (the joint's 3% price), the seal
at abab's seal_point 0.4 and the collapse at its push_limit 0.5 as density ratios
(1/0.6)**3 and (1/0.5)**3, lump_factor 2, leak 0.05 (abab's share per step, step_time
1). Chosen as units, not tuned: friction 1, clock_scale 1 (the start's mean
concentration), step_time 1, courant 0.4, potential_cycles 2, max_substeps 2000,
occupied_floor 0.001, cfl_floor 0.001 (the occupied floor's value).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    """Pull, push and its power, rest, seal, collapse, friction, leak, clocks, time stepping, thresholds."""

    attraction: float
    push: float
    push_power: float
    b_rest: float
    seal_compression: float
    collapse_compression: float
    friction: float
    leak: float
    clock_scale: float
    step_time: float
    courant: float
    potential_cycles: float
    max_substeps: float
    cfl_floor: float
    occupied_floor: float
    lump_factor: float

    @classmethod
    def all_on(cls, **overrides: float) -> "Settings":
        """Defaults with every rule on, plus the given overrides."""
        return replace(_ALL_ON, **overrides)


_ALL_ON = Settings(attraction=0.005, push=0.1, push_power=3.0, b_rest=0.03,
                   seal_compression=(1.0 / 0.6) ** 3, collapse_compression=(1.0 / 0.5) ** 3,
                   friction=1.0, leak=0.05, clock_scale=1.0, step_time=1.0, courant=0.4, potential_cycles=2.0,
                   max_substeps=2000.0, cfl_floor=1e-3, occupied_floor=1e-3, lump_factor=2.0)
