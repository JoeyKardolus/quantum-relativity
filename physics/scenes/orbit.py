"""Orbit scene: a small blob given sideways speed past a big one, inside the foam; radius, period, Kepler.

Contract: `orbit(cells_per_side=19, boxes_per_cell_side=6, distances=(10.0, 20.0),
big_energy=2000.0, big_radius=3.0, small_energy=20.0, small_radius=1.5, steps=120,
settings=None, device="cuda")` runs one start per distance (blob_start.py): an even
start of wrapped cells, a big ball of extra energy big_energy in the middle, a small
ball small_energy at that distance along x (energy volumes sit at all-even lattice places,
boxes_per_cell_side + 1 boxes apart, so the middle place is a volume when
(cells_per_side - 1) / 2 is odd, 7, 11, 15, 19, ..., and the balls land on energy
volumes when the distances are whole multiples of boxes_per_cell_side + 1), moving along y at the Newtonian circular
speed sqrt(G (M + m) / distance) for G = settings.pull_strength and the two balls' extra
quanta M, m (energies in units of blob_start's BOX_QUANTA, 1000 quanta), the big ball moving the other way so the pair's momentum is 0. Empty
space holds no boxes, so the blobs move through the foam (bible: the space between
masses is cells). Each run takes `steps` steps (default settings Settings(), every rule
on), measuring after every step. A blob is extra energy of whole energy volumes: a volume's
quanta above the even foam's volume (the median volume of the first sample), clipped at 0 and
laid evenly over its boxes. Jiggle inside a cell moves no energy between cells, so it is
no blob (box by box, the whole-number jiggle of 7 million boxes outweighed the small blob
within 8 steps, 2026-11-17). The big centre is the excess's centre within big_radius + 3
of its last place. The small blob is followed the same way: the excess within one cell
spacing (boxes_per_cell_side + 1) of its last centre and outside the big blob's reach. It
is lost (radius 0 from then on) when no excess is left there (fallen in or spread out) or
when that excess is over LOST_FACTOR (2) times its start's: the big blob's own spreading
has reached it and the two can no longer be told apart. A run's summary (mean radius,
turns, period, kepler) covers the samples up to the loss; `followed_time` is its last one.
Returns a dict (no verdict):
  cells_per_side, boxes_per_cell_side, steps, settings (dict), distances, big_energy,
  small_energy
  runs        per distance: time, radius (centre to centre, boxes), angle (unwrapped,
              radians, in the x-y plane), spread (RMS radius of the small blob's excess
              about its centre, rms_radius.py) and small_energy (its excess quanta) per
              sample, followed_time, mean_radius, min_radius, max_radius,
              turns (angle swept / 2 pi), period (2 pi / |least-squares angular speed|,
              inf when it does not turn), early_period (the same over the first quarter of
              the samples, at least 2), kepler (period**2 / mean_radius**3),
              angular_momentum per sample (the small blob's about the big centre:
              small_energy times the z part of offset x velocity, the velocity the
              offset's time derivative, central differences, one-sided at the ends; a
              central pull and frictionless flow keep it), angular_momentum_change
              (last over first, minus 1; NaN when the first is 0)
  kepler      kepler per distance; Kepler's rule holds when these are equal
  newton_kepler   4 pi**2 / (G (M + m)), the value Newton's pull gives every distance
                  (the pull settles at light speed, physics/engine/pull_wave.py, so a
                  blob farther than light_speed * time from the other has not felt it yet)
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.engine import Settings, State, cells, host_boxes
from physics.scenes.blob_start import BOX_QUANTA, blob_start
from physics.scenes.energy_centre import energy_centre
from physics.scenes.rms_radius import rms_radius
from physics.scenes.sampled_run import sampled_run

TRACK_MARGIN = 3.0
LOST_FACTOR = 2.0


def orbit(cells_per_side: int = 19, boxes_per_cell_side: int = 6, distances: tuple[float, ...] = (10.0, 20.0),
          big_energy: float = 2000.0,
          big_radius: float = 3.0, small_energy: float = 20.0, small_radius: float = 1.5, steps: int = 120,
          settings: Settings | None = None, device: str = "cuda") -> dict:
    """Radius, period and period**2 / radius**3 of a small blob round a big one, per distance."""
    settings = settings or Settings()
    pull = settings.pull_strength * (big_energy + small_energy) * BOX_QUANTA
    runs = []
    for distance in distances:
        speed = math.sqrt(pull / distance)
        universe = blob_start(cells_per_side, boxes_per_cell_side, [
            {"offset": (0.0, 0.0, 0.0), "radius": big_radius, "energy": big_energy,
             "velocity": (0.0, -speed * small_energy / big_energy, 0.0)},
            {"offset": (distance, 0.0, 0.0), "radius": small_radius, "energy": small_energy,
             "velocity": (0.0, speed, 0.0)}], device=device)
        middle = np.full(3, (universe.side - 1) / 2.0)
        tracker = _Tracker(middle, big_radius, middle + np.array([distance, 0.0, 0.0]), boxes_per_cell_side + 1.0)
        samples = sampled_run(universe, settings, steps, 1, tracker.measure)
        runs.append(_summary([time for time, _ in samples], [measured for _, measured in samples]))
    return {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "steps": steps,
            "settings": asdict(settings), "distances": list(distances),
            "big_energy": big_energy, "small_energy": small_energy, "runs": runs,
            "kepler": [run["kepler"] for run in runs], "newton_kepler": 4.0 * math.pi ** 2 / pull}


class _Tracker:
    """Follows both blobs from sample to sample; measure() gives the small blob's offset from the big one."""

    def __init__(self, big: np.ndarray, big_radius: float, small: np.ndarray, small_reach: float):
        self.big, self.reach = big, big_radius + TRACK_MARGIN
        self.small, self.small_reach = small, small_reach
        self.even = None
        self.first = None
        self.lost = False

    def measure(self, universe: State) -> dict:
        energy = self._excess(universe)
        axis = np.arange(universe.side, dtype=np.float64)
        self.big = energy_centre(energy, _distance(axis, self.big) <= self.reach)
        near = (_distance(axis, self.small) <= self.small_reach) & (_distance(axis, self.big) > self.reach)
        small = energy_centre(energy, near)
        held = float(energy[near].sum())
        if self.first is None:
            self.first = held
        if self.lost or np.isnan(small).any() or held > LOST_FACTOR * self.first:
            self.lost = True
            return {"offset": np.zeros(3), "spread": 0.0, "small_energy": 0.0}
        self.small = small
        return {"offset": small - self.big, "spread": rms_radius(energy, near),
                "small_energy": held}

    def _excess(self, universe: State) -> np.ndarray:
        """Per box, its energy volume's quanta above the even foam's volume, laid evenly over the volume."""
        found = cells(host_boxes(universe))
        count = found["volumes"]
        held = found["quanta"][:count].astype(np.float64)
        if self.even is None:
            self.even = float(np.median(held))
        per_box = np.maximum(held - self.even, 0.0) / np.maximum(found["boxes"][:count], 1)
        owner = found["owner"]
        in_volume = (owner >= 0) & (owner < count)
        return np.where(in_volume, per_box[np.clip(owner, 0, count - 1)], 0.0)


def _distance(axis: np.ndarray, centre: np.ndarray) -> np.ndarray:
    return np.sqrt((axis[:, None, None] - centre[0]) ** 2 + (axis[None, :, None] - centre[1]) ** 2
                   + (axis[None, None, :] - centre[2]) ** 2)


def _summary(times: list[float], measured: list[dict]) -> dict:
    energies = [sample["small_energy"] for sample in measured]
    followed = next((index for index, held in enumerate(energies) if held <= 0.0), len(energies))
    whole = {"time": list(times), "radius": [float(np.linalg.norm(sample["offset"])) for sample in measured],
             "spread": [sample["spread"] for sample in measured], "small_energy": energies,
             "followed_time": float(times[max(followed - 1, 0)])}
    if followed < 3:
        return whole | {"angle": [0.0] * len(times), "mean_radius": 0.0, "min_radius": 0.0, "max_radius": 0.0,
                        "turns": 0.0, "period": math.inf, "early_period": math.inf, "kepler": math.inf,
                        "angular_momentum": [0.0] * len(times), "angular_momentum_change": math.nan}
    times, measured = times[:followed], measured[:followed]
    offsets = np.array([sample["offset"] for sample in measured])
    radius = np.linalg.norm(offsets, axis=1)
    angle = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
    early = max(len(times) // 4, 2)
    period = _period(times, angle)
    mean_radius = float(radius.mean())
    angular = _angular_momentum(times, offsets, [sample["small_energy"] for sample in measured])
    pad = [0.0] * (len(energies) - followed)
    return whole | {"angle": angle.tolist() + pad, "mean_radius": mean_radius,
            "min_radius": float(radius.min()), "max_radius": float(radius.max()),
            "turns": float((angle[-1] - angle[0]) / (2.0 * math.pi)), "period": period,
            "early_period": _period(times[:early], angle[:early]),
            "kepler": period ** 2 / mean_radius ** 3 if mean_radius > 0 else math.inf,
            "angular_momentum": angular.tolist() + pad,
            "angular_momentum_change": float(angular[-1] / angular[0] - 1.0) if angular[0] != 0 else math.nan}


def _angular_momentum(times: list[float], offsets: np.ndarray, energies: list[float]) -> np.ndarray:
    """The small blob's angular momentum about the big centre along z, per sample."""
    velocity = np.gradient(offsets, np.asarray(times, dtype=np.float64), axis=0)
    return np.asarray(energies, dtype=np.float64) * (offsets[:, 0] * velocity[:, 1] - offsets[:, 1] * velocity[:, 0])


def _period(times: list[float], angle: np.ndarray) -> float:
    angular_speed, _ = np.polyfit(times, angle, 1)
    return 2.0 * math.pi / abs(angular_speed) if angular_speed != 0.0 else math.inf
