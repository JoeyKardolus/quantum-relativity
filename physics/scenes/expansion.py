"""Expansion scene: does the occupied region grow, and does its growth slow, then speed up?

Contract: `expansion(cells_per_side=7, boxes_per_cell_side=10, seed=1, unevenness=0.3,
steps=60, every=2, settings=None, device="cuda", joint_unevenness=0.0)` builds
physics.engine.new_universe(cells_per_side, boxes_per_cell_side, seed, unevenness,
joint_unevenness), runs
`steps` steps with settings (default Settings(), every rule on), measuring at the start and
every `every` steps (sampled_run.py). Returns a dict (numbers and lists of floats, no
verdict):
  cells_per_side, boxes_per_cell_side, seed, unevenness, joint_unevenness, steps, settings (dict)
  time               sample times
  occupied_growth    stats "occupied_growth" per sample: boxes holding space over the start's
  occupied_radius    radius of a ball of the occupied boxes (stats "occupied"), boxes
  energy_radius      RMS radius of all whole quanta (both signs) about their centre
                     (rms_radius.py), boxes
  edge_share         stats "edge_share" per sample (not ~0: no longer open space)
  rate_time          midpoints between samples
  occupied_rate      d ln(occupied_radius)/dt there (growth_rates.py)
  energy_rate        d ln(energy_radius)/dt there
  occupied_rate_trend_early, occupied_rate_trend_late   slope of occupied_rate against
                     time over the first and the second half of rate_time (least
                     squares); negative then positive = slowed, then sped up. NaN with
                     < 2 points.
  collapses, energy_drift   stats at the end (joint boxes collapsed; drift in quanta, 0
                     is exact)
  substeps           steps the run took (the engine has no substeps: one per step)
The Planck boxes are fixed; space grows as joints claim empty boxes at the edge, so the
occupied radius is the size of space and the energy radius says where the energy is.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.engine import Settings, State, new_universe, stats
from physics.scenes.box_energies import box_energies
from physics.scenes.growth_rates import growth_rates
from physics.scenes.rms_radius import rms_radius
from physics.scenes.sampled_run import sampled_run


def expansion(cells_per_side: int = 7, boxes_per_cell_side: int = 10, seed: int = 1, unevenness: float = 0.3,
              steps: int = 60, every: int = 2, settings: Settings | None = None, device: str = "cuda", joint_unevenness: float = 0.0) -> dict:
    """Occupied and energy radius over time, their growth rates and whether the rate turns."""
    settings = settings or Settings()
    universe = new_universe(cells_per_side, boxes_per_cell_side, seed, unevenness, device=device,
                            joint_unevenness=joint_unevenness)
    samples = sampled_run(universe, settings, steps, every, lambda sampled: _measure(sampled, settings))
    times = [time for time, _ in samples]
    occupied_radius = [measured["occupied_radius"] for _, measured in samples]
    energy_radius = [measured["energy_radius"] for _, measured in samples]
    rate_time, occupied_rate = growth_rates(times, occupied_radius)
    _, energy_rate = growth_rates(times, energy_radius)
    half = rate_time.size // 2
    final = stats(universe, settings)
    return {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "seed": seed,
            "unevenness": unevenness, "joint_unevenness": joint_unevenness, "steps": steps, "settings": asdict(settings), "time": times,
            "occupied_growth": [measured["occupied_growth"] for _, measured in samples],
            "occupied_radius": occupied_radius, "energy_radius": energy_radius,
            "edge_share": [measured["edge_share"] for _, measured in samples],
            "rate_time": rate_time.tolist(), "occupied_rate": occupied_rate.tolist(),
            "energy_rate": energy_rate.tolist(),
            "occupied_rate_trend_early": _trend(rate_time[:half], occupied_rate[:half]),
            "occupied_rate_trend_late": _trend(rate_time[half:], occupied_rate[half:]),
            "collapses": final["collapses"], "energy_drift": final["energy_drift"],
            "substeps": universe.step}


def _measure(universe: State, settings: Settings) -> dict:
    now = stats(universe, settings)
    a_energy, b_energy = box_energies(universe)
    return {"occupied_growth": now["occupied_growth"],
            "occupied_radius": (3.0 * now["occupied"] / (4.0 * math.pi)) ** (1.0 / 3.0),
            "energy_radius": rms_radius(a_energy + b_energy), "edge_share": now["edge_share"]}


def _trend(times: np.ndarray, rates: np.ndarray) -> float:
    if times.size < 2:
        return float("nan")
    slope, _ = np.polyfit(times, rates, 1)
    return float(slope)
