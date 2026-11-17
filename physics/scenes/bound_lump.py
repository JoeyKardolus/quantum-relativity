"""Bound lump scene: does the most packed energy volume keep its size while the rest of space changes?

Contract: `bound_lump(cells_per_side=16, seed=1, unevenness=0.1, steps=200, every=10,
settings=None, device="cuda", start=None)` builds physics.engine.new_universe(cells_per_side, seed,
unevenness; `start` passes other start amounts on), runs `steps` steps with settings (default Settings(), every rule on) and
keeps every cell's length at the start and every `every` steps. At the end the lump is
the energy volume holding the most quanta that has lived since the start (a cell id
is the same cell for good). Returns a dict (no verdict):
  cells_per_side, seed, unevenness, steps, settings (dict), time
  lump             its cell id; lump_quanta_factor its quanta over the start's mean
                   volume quanta at the end; is_lump whether that reaches LUMP_FACTOR
                   (physics.engine.stats, 2)
  lump_length      its length per sample; lump_growth its last length over its first
  rest_length      mean length of the other energy volumes living then, per sample;
                   rest_growth last over first
  size_growth      the universe's size at the end over the start's
  known            a bound object keeps its size while space grows: lump_growth 1 with
                   rest_growth above 1
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import ENERGY_VOLUME, Settings, new_universe, stats
from physics.engine.stats import LUMP_FACTOR
from physics.scenes.sampled_run import sampled_run


def bound_lump(cells_per_side: int = 16, seed: int = 1, unevenness: float = 0.1, steps: int = 200, every: int = 10,
               settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """The most packed volume's length against the other volumes' mean length, over time."""
    settings = settings or Settings()
    universe = new_universe(cells_per_side, seed, **{"unevenness": unevenness, **(start or {})}, device=device)
    first = universe.lattice
    samples = sampled_run(universe, settings, steps, every,
                          lambda sampled: (sampled.host("length")[:first].astype(np.float64),
                                           sampled.kind_h[:first] == ENERGY_VOLUME))
    held = universe.host("cell_quanta")[:first].astype(np.float64)
    held[universe.kind_h[:first] != ENERGY_VOLUME] = 0.0
    lump = int(np.argmax(held))
    lump_length = [float(length[lump]) for _, (length, _) in samples]
    rest_length = []
    for _, (length, living) in samples:
        others = living.copy()
        others[lump] = False
        rest_length.append(float(length[others].mean()))
    factor = float(held[lump] / universe.start_quanta[ENERGY_VOLUME])
    return {"cells_per_side": cells_per_side, "seed": seed, "unevenness": unevenness, "steps": steps,
            "settings": asdict(settings), "time": [time for time, _ in samples], "lump": lump,
            "lump_quanta_factor": factor, "is_lump": int(factor >= LUMP_FACTOR), "lump_length": lump_length,
            "lump_growth": lump_length[-1] / lump_length[0], "rest_length": rest_length,
            "rest_growth": rest_length[-1] / rest_length[0],
            "size_growth": stats(universe, settings)["size_growth"]}
