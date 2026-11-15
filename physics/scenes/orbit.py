"""Orbit scene: a small blob given sideways speed past a big one; radius, period, Kepler.

Contract: `orbit(side=64, distances=(10.0, 14.0), big_energy=2000.0, big_radius=3.0,
small_energy=20.0, small_radius=1.5, steps=120, settings=None, device="cuda")` runs
one start per distance (blob_start.py): the big ball of A-energy big_energy in the middle,
the small ball small_energy at that distance along x, moving along y at the Newtonian
circular speed sqrt(G (M + m) / distance) for G = settings.attraction and the two balls'
A-energies M, m (only A pulls), the big ball moving the other way so the pair's momentum
is 0. Each run takes `steps` steps (default settings Settings.all_on()), measuring after
every step. The big centre is the energy centre of the boxes within big_radius + 3 of
its last place; the small blob is all energy farther than that from the new big centre
(the fluid smears a small blob along its path, so a window round it loses it), and its
centre is that energy's centre; with nothing left there it has fallen in (radius 0).
Returns a dict (no verdict):
  side, steps, settings (dict), distances, big_energy, small_energy
  runs        per distance: time, radius (centre to centre, boxes), angle (unwrapped,
              radians, in the x-y plane), spread (RMS radius of the small blob's energy
              about its centre, rms_radius.py) and small_energy (its A- plus B-energy) per
              sample, mean_radius, min_radius, max_radius,
              turns (angle swept / 2 pi), period (2 pi / |least-squares angular speed|,
              inf when it does not turn), early_period (the same over the first quarter of
              the samples, at least 2, before the fluid has smeared the blob far),
              kepler (period**2 / mean_radius**3)
  kepler      kepler per distance; Kepler's rule holds when these are equal
  newton_kepler   4 pi**2 / (G (M + m)), the value Newton's pull gives every distance
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.fluid import Grid, Settings
from physics.scenes.blob_start import blob_start
from physics.scenes.box_energies import box_energies
from physics.scenes.energy_centre import energy_centre
from physics.scenes.rms_radius import rms_radius
from physics.scenes.sampled_run import sampled_run

TRACK_MARGIN = 3.0


def orbit(side: int = 64, distances: tuple[float, ...] = (10.0, 14.0), big_energy: float = 2000.0,
          big_radius: float = 3.0, small_energy: float = 20.0, small_radius: float = 1.5, steps: int = 120,
          settings: Settings | None = None, device: str = "cuda") -> dict:
    """Radius, period and period**2 / radius**3 of a small blob round a big one, per distance."""
    settings = settings or Settings.all_on()
    pull = settings.attraction * (big_energy + small_energy)
    runs = []
    for distance in distances:
        speed = math.sqrt(pull / distance)
        middle = (side - 1) / 2.0
        grid = blob_start(side, [
            {"centre": (middle, middle, middle), "radius": big_radius, "energy": big_energy,
             "velocity": (0.0, -speed * small_energy / big_energy, 0.0)},
            {"centre": (middle + distance, middle, middle), "radius": small_radius, "energy": small_energy,
             "velocity": (0.0, speed, 0.0)}], device=device)
        tracker = _Tracker(np.full(3, middle), big_radius)
        samples = sampled_run(grid, settings, steps, 1, tracker.measure)
        runs.append(_summary([time for time, _ in samples], [measured for _, measured in samples]))
    return {"side": side, "steps": steps, "settings": asdict(settings), "distances": list(distances),
            "big_energy": big_energy, "small_energy": small_energy, "runs": runs,
            "kepler": [run["kepler"] for run in runs], "newton_kepler": 4.0 * math.pi ** 2 / pull}


class _Tracker:
    """Follows the big centre from sample to sample; measure() gives the small blob's offset."""

    def __init__(self, big: np.ndarray, big_radius: float):
        self.big, self.reach = big, big_radius + TRACK_MARGIN

    def measure(self, grid: Grid) -> dict:
        a_energy, b_energy = box_energies(grid)
        energy = a_energy + b_energy
        axis = np.arange(grid.side, dtype=np.float64)
        self.big = energy_centre(energy, _distance(axis, self.big) <= self.reach)
        outside = _distance(axis, self.big) > self.reach
        small = energy_centre(energy, outside)
        if np.isnan(small).any():
            return {"offset": np.zeros(3), "spread": 0.0, "small_energy": 0.0}
        return {"offset": small - self.big, "spread": rms_radius(energy, outside),
                "small_energy": float(energy[outside].sum())}


def _distance(axis: np.ndarray, centre: np.ndarray) -> np.ndarray:
    return np.sqrt((axis[:, None, None] - centre[0]) ** 2 + (axis[None, :, None] - centre[1]) ** 2
                   + (axis[None, None, :] - centre[2]) ** 2)


def _summary(times: list[float], measured: list[dict]) -> dict:
    offsets = np.array([sample["offset"] for sample in measured])
    radius = np.linalg.norm(offsets, axis=1)
    angle = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
    early = max(len(times) // 4, 2)
    period = _period(times, angle)
    mean_radius = float(radius.mean())
    return {"time": list(times), "radius": radius.tolist(), "angle": angle.tolist(),
            "spread": [sample["spread"] for sample in measured],
            "small_energy": [sample["small_energy"] for sample in measured], "mean_radius": mean_radius,
            "min_radius": float(radius.min()), "max_radius": float(radius.max()),
            "turns": float((angle[-1] - angle[0]) / (2.0 * math.pi)), "period": period,
            "early_period": _period(times[:early], angle[:early]),
            "kepler": period ** 2 / mean_radius ** 3 if mean_radius > 0 else math.inf}


def _period(times: list[float], angle: np.ndarray) -> float:
    angular_speed, _ = np.polyfit(times, angle, 1)
    return 2.0 * math.pi / abs(angular_speed) if angular_speed != 0.0 else math.inf
