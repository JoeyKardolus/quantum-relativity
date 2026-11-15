"""Expansion scene: does the filled cube grow, and does its growth slow, then speed up?

Contract: `expansion(side=64, seed=1, unevenness=0.3, steps=60, every=2, settings=None,
device="cuda")` builds physics.fluid.new_grid(side, seed, unevenness), runs `steps` steps
with settings (default Settings.all_on()), measuring at the start and every `every` steps
(sampled_run.py). Returns a dict (numbers and lists of floats, no verdict):
  side, seed, unevenness, steps, settings (dict)
  time               sample times
  occupied_radius    radius of a ball of the occupied boxes (stats "occupied"), boxes
  energy_radius      RMS radius of all energy about its centre (rms_radius.py), boxes
  edge_share         stats "edge_share" per sample (not ~0: no longer open space)
  rate_time          midpoints between samples
  occupied_rate      d ln(occupied_radius)/dt there (growth_rates.py)
  energy_rate        d ln(energy_radius)/dt there
  energy_rate_trend_early, energy_rate_trend_late   slope of energy_rate against time
                     over the first and the second half of rate_time (least squares);
                     negative then positive = slowed, then sped up. NaN with < 2 points.
  energy_drift       stats "energy_drift" at the end (quanta; 0 is exact)
  substeps           substeps the run took
Box units throughout: the engine has fixed boxes and no length that follows energy yet
(bible rule 4), so this measures where the energy is, not a grown space.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.fluid import Grid, Settings, new_grid, stats
from physics.scenes.box_energies import box_energies
from physics.scenes.growth_rates import growth_rates
from physics.scenes.rms_radius import rms_radius
from physics.scenes.sampled_run import sampled_run


def expansion(side: int = 64, seed: int = 1, unevenness: float = 0.3, steps: int = 60, every: int = 2,
              settings: Settings | None = None, device: str = "cuda") -> dict:
    """Occupied and energy radius over time, their growth rates and whether the rate turns."""
    settings = settings or Settings.all_on()
    grid = new_grid(side, seed, unevenness, device=device)
    samples = sampled_run(grid, settings, steps, every, lambda sampled: _measure(sampled, settings))
    times = [time for time, _ in samples]
    occupied_radius = [measured["occupied_radius"] for _, measured in samples]
    energy_radius = [measured["energy_radius"] for _, measured in samples]
    rate_time, occupied_rate = growth_rates(times, occupied_radius)
    _, energy_rate = growth_rates(times, energy_radius)
    half = rate_time.size // 2
    final = stats(grid, settings)
    return {"side": side, "seed": seed, "unevenness": unevenness, "steps": steps, "settings": asdict(settings),
            "time": times, "occupied_radius": occupied_radius, "energy_radius": energy_radius,
            "edge_share": [measured["edge_share"] for _, measured in samples],
            "rate_time": rate_time.tolist(), "occupied_rate": occupied_rate.tolist(),
            "energy_rate": energy_rate.tolist(),
            "energy_rate_trend_early": _trend(rate_time[:half], energy_rate[:half]),
            "energy_rate_trend_late": _trend(rate_time[half:], energy_rate[half:]),
            "energy_drift": final["energy_drift"], "substeps": final["substeps"]}


def _measure(grid: Grid, settings: Settings) -> dict:
    now = stats(grid, settings)
    a_energy, b_energy = box_energies(grid)
    return {"occupied_radius": (3.0 * now["occupied"] / (4.0 * math.pi)) ** (1.0 / 3.0),
            "energy_radius": rms_radius(a_energy + b_energy), "edge_share": now["edge_share"]}


def _trend(times: np.ndarray, rates: np.ndarray) -> float:
    if times.size < 2:
        return float("nan")
    slope, _ = np.polyfit(times, rates, 1)
    return float(slope)
