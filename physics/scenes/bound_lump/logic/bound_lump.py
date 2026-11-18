"""Bound lump scene: does the most packed energy volume keep its size while the rest of space changes?

Contract: `bound_lump(cells_per_side=16, seed=1, steps=200, every=10, settings=None,
device="cuda", start=None)` builds physics.engine.new_universe with the settings
(default Settings(); `start` puts a candidate's start amounts in), runs `steps` steps
with every rule on and keeps every start cell's length at the start and every `every`
steps. At the end the lump is the energy volume holding the most quanta that has lived
since the start (a cell id is the same cell for good). Returns a dict (no verdict):
  cells_per_side, seed, steps, settings (dict), time
  lump             its cell id; lump_quanta_factor its quanta at the end over an even
                   start volume's; is_lump whether that reaches LUMP_FACTOR (2)
  lump_length      its length per sample; lump_growth its last length over its first
  rest_length      mean length of the other start energy volumes living and holding
                   energy then, per sample; rest_growth last over first
  size_growth      the universe's size at the end over the start's
  known            a bound object keeps its size while space grows: lump_growth 1 with
                   rest_growth above 1
Raises ValueError when no start energy volume lives at the end.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import ENERGY_VOLUME, Grid, Settings, measure_grid, new_universe
from physics.scenes.run.definitions.even_volume_quanta import even_volume_quanta
from physics.scenes.run.definitions.start_settings import start_settings
from physics.scenes.run.logic.sampled_run import sampled_run
from physics.scenes.volumes.definitions.lump_factor import LUMP_FACTOR


def _lengths(universe: Grid) -> tuple:
    """Every start cell's length, which of them are energy volumes holding energy, and the universe's size."""
    lattice = universe.side ** 3
    holding = (universe.host("kind")[:lattice] == ENERGY_VOLUME) & (universe.host("cell_quanta")[:lattice] > 0)
    return universe.host("length")[:lattice].astype(np.float64), holding, measure_grid(universe)["size"]


def _rest_mean(length: np.ndarray, holding: np.ndarray, lump: int) -> float:
    """Mean length of the holding volumes other than the lump (NaN for none)."""
    others = holding.copy()
    others[lump] = False
    return float(length[others].mean()) if others.any() else float("nan")


def bound_lump(cells_per_side: int = 16, seed: int = 1, steps: int = 200, every: int = 10,
               settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """The most packed volume's length against the other volumes' mean length, over time."""
    settings = start_settings(settings, start)
    universe = new_universe(settings, cells_per_side, seed, device=device)
    samples = sampled_run(universe, settings, steps, every, _lengths)
    lattice = universe.side ** 3
    held = np.where(universe.host("kind")[:lattice] == ENERGY_VOLUME, universe.host("cell_quanta")[:lattice], 0)
    if not held.any():
        raise ValueError("no start energy volume lives at the end: there is no lump to follow")
    lump = int(np.argmax(held))
    lump_length = [float(length[lump]) for _, (length, _, _) in samples]
    rest_length = [_rest_mean(length, holding, lump) for _, (length, holding, _) in samples]
    factor = float(held[lump]) / even_volume_quanta(settings)
    return {"cells_per_side": cells_per_side, "seed": seed, "steps": steps, "settings": asdict(settings),
            "time": [time for time, _ in samples], "lump": lump, "lump_quanta_factor": factor,
            "is_lump": int(factor >= LUMP_FACTOR), "lump_length": lump_length,
            "lump_growth": lump_length[-1] / lump_length[0], "rest_length": rest_length,
            "rest_growth": rest_length[-1] / rest_length[0],
            "size_growth": samples[-1][1][2] / samples[0][1][2]}
