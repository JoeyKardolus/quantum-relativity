"""Two lumps scene: does energy attract energy, and how does the pull between two lumps fall with distance?

Contract: `two_lumps(cells_per_side=48, apart=(2, 3, 4, 6), factor=4.0, steps=1200,
settle=600, seed=1, settings=None, start=None, device="cuda")` runs, for every distance in
`apart` (energy volumes from lump to lump along one lattice axis), a placed start
(packed_start: an even grid with two volumes packed `factor` times; `start` passes the
fitted start amounts on) for `steps` steps with every rule on, and beside it the same
start with only the first lump. Returns a dict (numbers and lists, no verdict):
  cells_per_side, factor, steps, settle, seed, settings (dict), apart
  distance      per run, the summed lengths of the cells between the two lumps' middles
                at the start (the bible's distance)
  tilt          per run, how much deeper the pull lies on the second lump's side of
                the first: (pull at the volume beside it away from the second - pull
                at the volume beside it toward the second) / the summed lengths between
                those two volumes' middles, the mean over the steps from `settle` on
                (the pull is a wave that keeps swinging). Above 0: the pull draws the
                first lump's energy toward the second
  tilt_alone    the same number at the same place with the second lump not there: what
                the even grid's own small unevenness gives
  pull          tilt - tilt_alone: the second lump's own pull at the first
  newton        what a pull of one over distance squared gives for the same
                measurement in the same wrapping box, as a ratio to its value at the
                first distance: the potential 1 / distance of the other lump and its
                copies in the neighbouring boxes (4 each way; a box is the summed
                lengths once round the lumps' row), differenced over the same two
                volumes beside the lump. pull / pull[0] is read beside it
  pull_power, newton_power   least-squares slope of ln(pull) and ln(newton) against
                ln(distance) (NaN unless every pull is above 0)
  closing       per run, how far the two lumps' extra energy moved toward each other
                over the run, in energy volumes: for each lump the quanta-weighted
                mean place along the axis of the energy above an even volume's, over
                the volumes within apart // 2 of it (a tube 5 volumes wide), end minus
                start, the pair's sum; closing_alone the first lump's own shift with
                the second not there (about 0)
  kept          per run, the share of the lumps' extra energy still in the two lump
                volumes at the end
Raises ValueError for a distance under 2 or over a third of the volumes a side.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import ENERGY_VOLUME, Settings, State, step
from physics.scenes.packed_start import packed_start
from physics.scenes.volume_field import volume_field

_TUBE = 2


def two_lumps(cells_per_side: int = 48, apart: tuple = (2, 3, 4, 6), factor: float = 4.0, steps: int = 1200,
              settle: int = 600, seed: int = 1, settings: Settings | None = None, start: dict | None = None,
              device: str = "cuda") -> dict:
    """The pull each of two placed lumps feels toward the other at several distances, and whether they close in."""
    settings = settings or Settings()
    across = cells_per_side // 2
    if min(apart) < 2 or max(apart) > across // 3:
        raise ValueError(f"distances must be 2 .. {across // 3} volumes, got {apart}")
    runs = [_run(cells_per_side, gap, factor, steps, settle, seed, settings, start, device) for gap in apart]
    pull = np.array([run["tilt"] - run["tilt_alone"] for run in runs])
    distance = np.array([run["distance"] for run in runs])
    newton = np.array([_newton(run["distance"], run["reach"], run["round"]) for run in runs])
    newton = newton / newton[0]
    return {"cells_per_side": cells_per_side, "factor": factor, "steps": steps, "settle": settle, "seed": seed,
            "settings": asdict(settings), "apart": list(apart), "distance": distance.tolist(),
            "tilt": [run["tilt"] for run in runs], "tilt_alone": [run["tilt_alone"] for run in runs],
            "pull": pull.tolist(), "newton": newton.tolist(),
            "pull_power": _power(distance, pull), "newton_power": _power(distance, newton),
            "closing": [run["closing"] for run in runs], "closing_alone": [run["closing_alone"] for run in runs],
            "kept": [run["kept"] for run in runs]}


def _run(cells: int, gap: int, factor: float, steps: int, settle: int, seed: int, settings: Settings,
         start: dict | None, device: str) -> dict:
    """One distance: the pair and the first lump alone, stepped side by side in time."""
    middle = (cells // 4) * 2
    low = middle - 2 * (gap // 2)
    places = [(low, middle, middle), (low + 2 * gap, middle, middle)]
    pair = packed_start(cells, {place: factor for place in places}, seed, settings, device, start)
    alone = packed_start(cells, {places[0]: factor}, seed, settings, device, start)
    even = pair.start_quanta[ENERGY_VOLUME]
    distance = _between(pair, places[0][0], places[1][0], middle)
    before = [_centre(universe, place, gap, even) for universe, place in ((pair, places[0]), (pair, places[1]),
                                                                          (alone, places[0]))]
    tilts, lone = [], []
    for taken in range(1, steps + 1):
        step(pair, settings)
        step(alone, settings)
        if taken > settle:
            tilts.append(_tilt(pair, places[0], 1))
            lone.append(_tilt(alone, places[0], 1))
    after = [_centre(universe, place, gap, even) for universe, place in ((pair, places[0]), (pair, places[1]),
                                                                         (alone, places[0]))]
    held = volume_field(pair)[0]
    extra = sum(float(held[place[0] // 2, place[1] // 2, place[2] // 2]) - even for place in places)
    return {"distance": distance, "reach": 0.5 * _between(pair, places[0][0] - 2, places[0][0] + 2, middle),
            "round": _between(pair, 0, cells, middle), "tilt": float(np.mean(tilts)), "tilt_alone": float(np.mean(lone)),
            "closing": float((after[0] - before[0]) - (after[1] - before[1])),
            "closing_alone": float(after[2] - before[2]), "kept": extra / (2.0 * (factor - 1.0) * even)}


def _cell(universe: State, first: int, second: int, third: int) -> int:
    """The id of the lattice place (wrapping)."""
    side = universe.side
    return ((first % side) * side + second % side) * side + third % side


def _between(universe: State, low: int, high: int, middle: int) -> float:
    """Summed lengths of the cells from the middle of place `low` to the middle of place `high` along the axis."""
    length = universe.host("length").astype(np.float64)
    row = [length[_cell(universe, index, middle, middle)] for index in range(low, high + 1)]
    return float(sum(row) - 0.5 * (row[0] + row[-1]))


def _tilt(universe: State, place: tuple, toward: int) -> float:
    """How much deeper the pull lies on the `toward` side of a lump, per length between its two neighbour volumes."""
    pull = universe.host("pull")
    near = _cell(universe, place[0] + 2 * toward, place[1], place[2])
    far = _cell(universe, place[0] - 2 * toward, place[1], place[2])
    return float(pull[far] - pull[near]) / _between(universe, place[0] - 2, place[0] + 2, place[1])


def _centre(universe: State, place: tuple, gap: int, even: float) -> float:
    """Mean place along the axis, in volumes from the lump, of the energy above an even volume's near the lump."""
    held = volume_field(universe)[0]
    across = held.shape[0]
    reach = gap // 2
    offsets = np.arange(-reach, reach + 1)
    tube = np.arange(-_TUBE, _TUBE + 1)
    block = held[np.ix_((place[0] // 2 + offsets) % across, (place[1] // 2 + tube) % across,
                        (place[2] // 2 + tube) % across)]
    extra = np.maximum(block - even, 0.0).sum(axis=(1, 2))
    return float((extra * offsets).sum() / extra.sum()) if extra.sum() > 0.0 else 0.0


def _newton(distance: float, reach: float, box: float) -> float:
    """The tilt a 1 / distance potential gives over `reach` either side, from a point `distance` away and its copies."""
    shifts = np.arange(-4, 5) * box
    east, north, up = np.meshgrid(shifts, shifts, shifts, indexing="ij")
    depth = [(1.0 / np.sqrt((distance + offset + east) ** 2 + north ** 2 + up ** 2)).sum() for offset in (-reach, reach)]
    return float((depth[0] - depth[1]) / (2.0 * reach))


def _power(distance: np.ndarray, value: np.ndarray) -> float:
    """Least-squares slope of ln(value) against ln(distance); NaN unless every value is above 0."""
    if len(value) < 2 or np.any(value <= 0.0):
        return float("nan")
    return float(np.polyfit(np.log(distance), np.log(value), 1)[0])
