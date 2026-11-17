"""Timescape scene: does the universe's growth look different timed by clocks in the most packed volumes?

Contract: `timescape(cells_per_side=16, seed=1, unevenness=0.1, steps=200, every=10,
observers=8, settings=None, device="cuda")` builds physics.engine.new_universe(
cells_per_side, seed, unevenness), runs `steps` steps with settings (default
Settings(), every rule on) and keeps, at the start and every `every` steps, the
universe's size (stats "size") and every cell's own clock (state.clock: time is local,
a packed cell's clock runs slow). The observers are the `observers` energy volumes
holding the most quanta at the end that lived all run (observer_cells.py); the empty
side is the same number holding the least. Timescape cosmology (Wiltshire): expansion
only looks like it speeds up because clocks in dense regions, where it is measured,
run slower than in the voids. Returns a dict (no verdict):
  cells_per_side, seed, unevenness, steps, observers, settings (dict)
  time             the step counter's time per sample
  observer_time    the observers' mean clock per sample; void_time the emptiest
                   volumes' mean clock per sample
  size             the universe's size per sample
  q_counter, q_observer, q_void   the deceleration q (deceleration.py; q < 0 speeds up)
                   of size against each of the three times
  clock_ratio      observer_time over void_time at the end (below 1: dense clocks slow)
  known            real supernovae give q about -0.55 today
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import ENERGY_VOLUME, Settings, new_universe, stats
from physics.scenes.deceleration import deceleration
from physics.scenes.observer_cells import observer_cells
from physics.scenes.sampled_run import sampled_run

KNOWN_Q = -0.55


def timescape(cells_per_side: int = 16, seed: int = 1, unevenness: float = 0.1, steps: int = 200, every: int = 10,
              observers: int = 8, settings: Settings | None = None, device: str = "cuda") -> dict:
    """The universe's deceleration by the counter, by the densest volumes' clocks and by the emptiest's."""
    settings = settings or Settings()
    universe = new_universe(cells_per_side, seed, unevenness, device=device)
    first = universe.lattice
    samples = sampled_run(universe, settings, steps, every,
                          lambda sampled: (stats(sampled, settings)["size"],
                                           sampled.host("clock")[:first].astype(np.float64)))
    lived = universe.kind_h[:first] == ENERGY_VOLUME
    ranked = [int(cell) for cell in observer_cells(universe, universe.count) if cell < first and lived[cell]]
    dense, empty = ranked[:observers], ranked[-observers:]
    times = [time for time, _ in samples]
    size = [measured[0] for _, measured in samples]
    observer_time = [float(measured[1][dense].mean()) for _, measured in samples]
    void_time = [float(measured[1][empty].mean()) for _, measured in samples]
    return {"cells_per_side": cells_per_side, "seed": seed, "unevenness": unevenness, "steps": steps,
            "observers": len(dense), "settings": asdict(settings), "time": times, "observer_time": observer_time,
            "void_time": void_time, "size": size, "q_counter": deceleration(times, size),
            "q_observer": deceleration(observer_time, size), "q_void": deceleration(void_time, size),
            "clock_ratio": observer_time[-1] / void_time[-1], "known_q": KNOWN_Q}
