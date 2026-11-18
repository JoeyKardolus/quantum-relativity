"""Timescape scene: does the universe's growth look different timed by clocks in the most packed volumes?

Contract: `timescape(cells_per_side=16, seed=1, steps=200, every=10, observers=8,
settings=None, device="cuda", start=None)` builds physics.engine.new_universe with the
settings (default Settings(); `start` puts a candidate's start amounts in), runs `steps`
steps with every rule on and keeps, at the start and every `every` steps, the universe's
size (measure_grid) and every start cell's own clock (the engine's `clock`: time is
local, a packed cell's clock runs slow). The observers are the `observers` energy
volumes holding the most quanta at the end that lived all run (observer_cells); the
empty side is the same number holding the least. Returns a dict (no verdict):
  cells_per_side, seed, steps, observers (how many were used), settings (dict)
  time             the step counter's time per sample
  observer_time    the observers' mean clock per sample; void_time the emptiest
                   volumes' mean clock per sample
  size             the universe's size per sample
  q_counter, q_observer, q_void   the deceleration q (q < 0 speeds up) of size against
                   each of the three times
  clock_ratio      observer_time over void_time at the end (below 1: dense clocks slow)
  known_q          what real supernovae give, -0.55
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import Grid, Settings, measure_grid, new_universe
from physics.scenes.run.definitions.start_settings import start_settings
from physics.scenes.run.logic.sampled_run import sampled_run
from physics.scenes.timescape.definitions.deceleration import deceleration
from physics.scenes.timescape.definitions.known import KNOWN_Q
from physics.scenes.volumes.logic.observer_cells import observer_cells


def _size_and_clocks(universe: Grid) -> tuple:
    """The universe's size and the clock of every start cell now."""
    return measure_grid(universe)["size"], universe.host("clock")[:universe.side ** 3].astype(np.float64)


def _mean_clock(samples: list, cells: list) -> list:
    """Per sample, the mean clock of the named cells (NaN for none)."""
    return [float(clocks[cells].mean()) if cells else float("nan") for _, (_, clocks) in samples]


def timescape(cells_per_side: int = 16, seed: int = 1, steps: int = 200, every: int = 10, observers: int = 8,
              settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """The universe's deceleration by the counter, by the densest volumes' clocks and by the emptiest's."""
    settings = start_settings(settings, start)
    universe = new_universe(settings, cells_per_side, seed, device=device)
    samples = sampled_run(universe, settings, steps, every, _size_and_clocks)
    ranked = [int(cell) for cell in observer_cells(universe, universe.count) if cell < universe.side ** 3]
    dense, empty = ranked[:observers], ranked[-observers:] if observers > 0 else []
    times = [time for time, _ in samples]
    size = [measured[0] for _, measured in samples]
    observer_time, void_time = _mean_clock(samples, dense), _mean_clock(samples, empty)
    return {"cells_per_side": cells_per_side, "seed": seed, "steps": steps, "observers": len(dense),
            "settings": asdict(settings), "time": times, "observer_time": observer_time, "void_time": void_time,
            "size": size, "q_counter": deceleration(times, size),
            "q_observer": deceleration(observer_time, size) if dense else float("nan"),
            "q_void": deceleration(void_time, size) if empty else float("nan"),
            "clock_ratio": observer_time[-1] / void_time[-1] if dense and void_time[-1] else float("nan"),
            "known_q": KNOWN_Q}
