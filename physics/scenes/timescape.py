"""Timescape scene: does expansion look like it speeds up when timed by clocks inside a dense lump?

Bible `physics/quantum-relativity`, rule 8 ("Drift from imperfect transfer") and "How results
are trusted": an unfitted test. Timescape cosmology (Wiltshire) says expansion only looks
accelerating because astronomers time it with clocks in dense regions, which run at a different
rate from those in the emptying voids. Sources: Seifert, Lane, Galoppo, Ridden-Harper and
Wiltshire, MNRAS Letters 2025, doi 10.1093/mnrasl/slae112 (Pantheon+ supernovae favour
timescape); DES Collaboration, arXiv 2406.05048 (supernovae with other data combined disfavour it).

Contract: `timescape(cells_per_side=7, boxes_per_cell_side=10, seed=1, unevenness=0.3, steps=60,
every=2, observers=3, settings=None, device="cuda", joint_unevenness=0.0)` builds
physics.engine.new_universe(cells_per_side, boxes_per_cell_side, seed, unevenness,
joint_unevenness) (nothing
placed), runs it with settings (default Settings(), every rule on) keeping every energy
volume's own elapsed local time (clocked_run.py), and picks the `observers` most packed
energy volumes at the end (observer_cells.py). The observers' clock is the mean of their
elapsed local times. The size of
space is timed twice: (a) by the global step counter, (b) by the observers' clock. The first
sample after step 0 is the baseline (kept from the cell engine, whose first step was a size
correction); the growth and deceleration numbers use the samples from there on. Returns a dict
(numbers and lists of floats, no verdict):
  cells_per_side, boxes_per_cell_side, seed, unevenness, joint_unevenness, steps, observers, settings (dict)
  observer_ids        the observer volumes (physics.engine cells labels at the end), most
                      packed first
  observer_packing    their quanta per box at the end over the start's mean plus quanta per
                      plus box
  time                global counter time per sample
  local_time          the observers' mean elapsed local time per sample
  each_local_time     per observer, its own elapsed local time per sample
  clock_ratio         local_time / time per sample after the start (1: the observers' clock
                      kept the counter's pace)
  clock_mean          stats "clock_mean" (clock rate over the occupied boxes) per sample
  occupied_radius     radius of a ball of the occupied boxes, boxes (the size of space)
  energy_radius       RMS radius of all whole quanta about their centre, boxes
  edge_share          stats "edge_share" per sample (not ~0: no longer open space)
  global_rate, local_rate   d ln(occupied_radius)/dT between samples, T the counter time or
                      the observers' clock (growth_rates.py), from the baseline on
  global_q, local_q   deceleration q of occupied_radius against each time over the run
                      from the baseline (deceleration.py: q < 0 speeds up, q > 0 slows)
  global_q_late, local_q_late   the same over the later half of those samples
  energy_global_q, energy_local_q   q of energy_radius against each time
  each_local_q        q of occupied_radius against each observer's own clock
  collapses, energy_drift   stats at the end
NaN where a number cannot be formed (no observer, fewer than 3 samples for a q).
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.engine import Settings, State, cells, host_boxes, new_universe, stats
from physics.scenes.open_margin import open_margin
from physics.scenes.box_energies import box_energies
from physics.scenes.clocked_run import clocked_run
from physics.scenes.deceleration import deceleration
from physics.scenes.growth_rates import growth_rates
from physics.scenes.observer_cells import observer_cells
from physics.scenes.rms_radius import rms_radius


def timescape(cells_per_side: int = 7, boxes_per_cell_side: int = 10, seed: int = 1, unevenness: float = 0.3,
              steps: int = 60, every: int = 2, observers: int = 3, settings: Settings | None = None,
              device: str = "cuda", joint_unevenness: float = 0.0) -> dict:
    """Growth of space timed by the step counter and by clocks in the densest lumps, with both q's."""
    settings = settings or Settings()
    universe = new_universe(cells_per_side, boxes_per_cell_side, seed, unevenness, device=device,
                            margin=open_margin(cells_per_side, boxes_per_cell_side),
                            joint_unevenness=joint_unevenness)
    samples, local = clocked_run(universe, settings, steps, every, lambda sampled: _measure(sampled, settings))
    final = stats(universe, settings)
    picked = observer_cells(universe, observers, local)
    times = np.array([when for when, _ in samples])
    each_local = local[:, picked].T if picked.size else np.full((0, times.size), np.nan)
    local_time = each_local.mean(axis=0) if picked.size else np.full(times.size, np.nan)
    occupied = np.array([measured["occupied_radius"] for _, measured in samples])
    energy = np.array([measured["energy_radius"] for _, measured in samples])
    found = cells(host_boxes(universe))
    start = universe.start[0]
    rest = start["plus"] / max(start["plus_boxes"], 1)
    packing = found["quanta"][picked] / np.maximum(found["boxes"][picked], 1) / rest
    result = {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "seed": seed,
              "unevenness": unevenness, "joint_unevenness": joint_unevenness, "steps": steps, "observers": observers, "settings": asdict(settings),
              "observer_ids": picked.tolist(), "observer_packing": packing.tolist(), "time": times.tolist(),
              "local_time": local_time.tolist(), "each_local_time": each_local.tolist(),
              "clock_ratio": (local_time[1:] / times[1:]).tolist(),
              "clock_mean": [measured["clock_mean"] for _, measured in samples],
              "occupied_radius": occupied.tolist(), "energy_radius": energy.tolist(),
              "edge_share": [measured["edge_share"] for _, measured in samples],
              "global_rate": _rates(times[1:], occupied[1:]), "local_rate": _rates(local_time[1:], occupied[1:]),
              "collapses": final["collapses"], "energy_drift": final["energy_drift"]}
    result |= _decelerations(times[1:], local_time[1:], occupied[1:], energy[1:])
    result["each_local_q"] = [_q(own[1:], occupied[1:]) for own in each_local]
    return result


def _measure(universe: State, settings: Settings) -> dict:
    now = stats(universe, settings)
    a_energy, b_energy = box_energies(universe)
    return {"occupied_radius": (3.0 * now["occupied"] / (4.0 * math.pi)) ** (1.0 / 3.0),
            "energy_radius": rms_radius(a_energy + b_energy), "edge_share": now["edge_share"],
            "clock_mean": now["clock_mean"]}


def _rates(times: np.ndarray, sizes: np.ndarray) -> list:
    if times.size < 2 or not np.isfinite(times).all():
        return []
    return growth_rates(times, sizes)[1].tolist()


def _q(times: np.ndarray, sizes: np.ndarray) -> float:
    if not np.isfinite(times).all():
        return float("nan")
    return deceleration(times, sizes)


def _decelerations(global_time: np.ndarray, local_time: np.ndarray, occupied: np.ndarray,
                   energy: np.ndarray) -> dict:
    late = global_time.size // 2
    return {"global_q": _q(global_time, occupied), "local_q": _q(local_time, occupied),
            "global_q_late": _q(global_time[late:], occupied[late:]),
            "local_q_late": _q(local_time[late:], occupied[late:]),
            "energy_global_q": _q(global_time, energy), "energy_local_q": _q(local_time, energy)}
