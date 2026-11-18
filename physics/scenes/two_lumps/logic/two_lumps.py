"""Two lumps scene: does energy attract energy, and how does the pull between two lumps fall with distance?

Contract: `two_lumps(cells_per_side=48, apart=(2, 3, 4, 6), factor=4.0, steps=1200,
settle=600, seed=1, settings=None, start=None, device="cuda")` runs, for every distance
in `apart` (energy volumes from lump to lump along one lattice axis), a placed start
(pair_run: an even grid with two volumes packed `factor` times; `start` puts a
candidate's start amounts in) for `steps` steps with every rule on, and beside it the
same start with only the first lump. Returns a dict (numbers and lists, no verdict):
  cells_per_side, factor, steps, settle, seed, settings (dict), apart
  distance      per run, the summed lengths of the cells between the two lumps' middles
                at the start (the bible's distance)
  tilt, tilt_alone   per run (pair_run): the pull's tilt at the first lump toward the
                second, and the same with the second not there
  pull          tilt - tilt_alone: the second lump's own pull at the first. Above 0:
                it draws the first lump's energy toward the second
  newton        what a pull of one over distance squared gives for the same measurement
                in the same wrapping box (newton_tilt), as a ratio to its value at the
                first distance. pull / pull[0] is read beside it
  pull_power, newton_power   least-squares slope of ln(pull) and ln(newton) against
                ln(distance) (NaN unless every pull is above 0 and there are 2 or more)
  closing, closing_alone, kept   per run (pair_run)
Raises ValueError for a distance under 2 or over a third of the volumes a side, a
factor not above 1, or steps < 1.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import Settings
from physics.scenes.along_grid.definitions.log_slope import log_slope
from physics.scenes.run.definitions.start_settings import start_settings
from physics.scenes.two_lumps.definitions.newton_tilt import newton_tilt
from physics.scenes.two_lumps.logic.pair_run import pair_run

_FROM_RUN = ("tilt", "tilt_alone", "closing", "closing_alone", "kept")


def two_lumps(cells_per_side: int = 48, apart: tuple = (2, 3, 4, 6), factor: float = 4.0, steps: int = 1200,
              settle: int = 600, seed: int = 1, settings: Settings | None = None, start: dict | None = None,
              device: str = "cuda") -> dict:
    """The pull each of two placed lumps feels toward the other at several distances, and whether they close in."""
    across = cells_per_side // 2
    if not apart or min(apart) < 2 or max(apart) > across // 3:
        raise ValueError(f"distances must be 2 .. {across // 3} volumes, got {apart}")
    if not factor > 1.0 or steps < 1:
        raise ValueError(f"a lump holds more than 1 even volume and a run takes a step, got {factor} and {steps}")
    settings = start_settings(settings, start)
    runs = [pair_run(cells_per_side, gap, factor, steps, settle, seed, settings, device) for gap in apart]
    pull = np.array([run["tilt"] - run["tilt_alone"] for run in runs])
    distance = np.array([run["distance"] for run in runs])
    newton = np.array([newton_tilt(run["distance"], run["reach"], run["round"]) for run in runs])
    newton = newton / newton[0]
    return {"cells_per_side": cells_per_side, "factor": factor, "steps": steps, "settle": settle, "seed": seed,
            "settings": asdict(settings), "apart": list(apart), "distance": distance.tolist(),
            **{name: [run[name] for run in runs] for name in _FROM_RUN},
            "pull": pull.tolist(), "newton": newton.tolist(),
            "pull_power": log_slope(distance, pull), "newton_power": log_slope(distance, newton)}
