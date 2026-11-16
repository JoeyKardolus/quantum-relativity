"""Isotropy scene: is the foam the same in every direction? Pull and flow along a grid axis against a diagonal.

Contract: `isotropy(cells_per_side=12, boxes_per_cell_side=4, blob_energy=500.0,
blob_radius=3.0, flow_density=1.0, speed=0.3, steps=20, cone_degrees=20.0,
settings=None, device="cuda")` makes two measurements on blob_start's even
checkerboard (default settings Settings.all_on()). The Planck boxes form a cubic grid
with six face directions, so a direction-free foam should show no difference between a
face axis and a diagonal. Every ball sits on the middle energy volume: the one whose
box centre lies nearest the centre of all the energy volumes' boxes (with an even
cells_per_side the energy volumes, at all-odd cell places, are not centred on the grid),
so the foam's own pull and push are even round it.
Pull: one ball of extra energy blob_energy and radius blob_radius at rest, one step (the
pull is solved in it), then every living cell's mean pull potential (stats refreshes the
cell sums) against its centre's distance from the ball. A cell counts as "axis" when its
direction from the ball lies within cone_degrees of a face axis (plus or minus x, y, z),
"diagonal" when within cone_degrees of a body diagonal (plus or minus 1, 1, 1).
Distances are binned one cell side wide from blob_radius + 1 out to the farthest bin
both sets reach.
Flow: the ball of radius blob_radius holds flow_density extra energy per box (A and B
alike: a ball over whole energy volumes, not a spike in one) and is given `speed`
(boxes per time) along x, along the face diagonal (1, 1, 0) / sqrt 2 and along the body
diagonal (1, 1, 1) / sqrt 3, `steps` steps each, plus two runs of the same ball at rest.
What is followed is the centre of all the energy (energy_centre.py over every box,
energy volumes and joints), so energy that leaves the ball through the seams, as sound
or into joints, still counts; the first rest run's shift (the foam's own pull and push)
is taken off each moving run's. Its known value is where the start's momentum carries it:
the start's lattice momentum P = sum over box faces inside one cell of
Im(conj psi_a psi_b) along the face (the flow the hops make at openness 1), over the
total energy, times the time; a seam or a closed cell border turns or stops some of it,
which is what "carried" shows.
Returns a dict (no verdict; same in every direction means the ratios are 1 and the
differences 0):
  cells_per_side, boxes_per_cell_side, steps, settings (dict), blob_energy, flow_density,
  speed, ball_offset (the ball's place from the grid's middle, boxes)
  pull     distance (bin middles, boxes), axis, diagonal (mean cell potential per bin),
           axis_cells, diagonal_cells (cells per bin), relative_difference per bin
           ((diagonal - axis) / |axis|), mean_relative_difference (mean of its absolute)
  flow     per direction name ("axis", "face_diagonal", "body_diagonal"): along,
           across (boxes, the energy centre's shift less the rest run's), known (the
           start momentum's shift along the direction), carried (along / known);
           face_ratio and body_ratio (that diagonal's carried over the axis's);
           rest_shift (the first rest run's shift, 3 floats); rest_scatter (how far the
           two rest runs' shifts differ, boxes: the engine is not bit-repeatable, so a
           moving run's along or across smaller than this is not measured)
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.potts import ENERGY_VOLUME, Settings, new_universe, stats, step
from physics.scenes.blob_start import blob_start
from physics.scenes.energy_centre import energy_centre
from physics.scenes.wave_field import wave_field

DIRECTIONS = {"axis": (1.0, 0.0, 0.0), "face_diagonal": (1.0, 1.0, 0.0), "body_diagonal": (1.0, 1.0, 1.0)}


def isotropy(cells_per_side: int = 12, boxes_per_cell_side: int = 4, blob_energy: float = 500.0,
             blob_radius: float = 3.0, flow_density: float = 1.0, speed: float = 0.3, steps: int = 20,
             cone_degrees: float = 20.0, settings: Settings | None = None, device: str = "cuda") -> dict:
    """Pull profile along axes against diagonals, and how far a moving ball's energy goes along each."""
    settings = settings or Settings.all_on()
    offset = _middle_volume_offset(cells_per_side, boxes_per_cell_side, device)
    pull_blob = {"offset": offset, "radius": blob_radius, "energy": blob_energy}
    flow_blob = {"offset": offset, "radius": blob_radius, "density": flow_density}
    rest, again = (_shift(cells_per_side, boxes_per_cell_side, flow_blob, (0.0, 0.0, 0.0), steps, settings, device)[0]
                   for _ in range(2))
    flow = {}
    for name, direction in DIRECTIONS.items():
        unit = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
        shift, momentum = _shift(cells_per_side, boxes_per_cell_side, flow_blob, tuple(speed * unit), steps,
                                 settings, device)
        moved = shift - rest
        along = float(moved @ unit)
        known = float(momentum @ unit) * steps * settings.step_time
        flow[name] = {"along": along, "across": float(np.linalg.norm(moved - along * unit)), "known": known,
                      "carried": _ratio(along, known)}
    flow["face_ratio"] = _ratio(flow["face_diagonal"]["carried"], flow["axis"]["carried"])
    flow["body_ratio"] = _ratio(flow["body_diagonal"]["carried"], flow["axis"]["carried"])
    flow["rest_shift"] = [float(value) for value in rest]
    flow["rest_scatter"] = float(np.linalg.norm(again - rest))
    return {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "steps": steps,
            "settings": asdict(settings), "blob_energy": blob_energy, "flow_density": flow_density, "speed": speed,
            "ball_offset": [float(value) for value in offset],
            "pull": _pull_profile(cells_per_side, boxes_per_cell_side, pull_blob, cone_degrees, settings, device),
            "flow": flow}


def _middle_volume_offset(cells_per_side, boxes_per_cell_side, device) -> tuple:
    """The box centre of the energy volume nearest all energy volumes' centre, from the grid's middle."""
    even = new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0.0, device=device)
    owner, kind = even.owner.numpy(), even.kind.numpy()
    volume_box = (owner >= 0) & (kind[np.maximum(owner, 0)] == ENERGY_VOLUME)
    places = np.argwhere(volume_box).astype(np.float64)
    cells = owner[volume_box]
    boxes = np.bincount(cells, minlength=kind.size).astype(np.float64)
    centres = np.stack([np.bincount(cells, weights=places[:, axis], minlength=kind.size) for axis in range(3)],
                       axis=1)
    volumes = np.flatnonzero(boxes > 0)
    centres = centres[volumes] / boxes[volumes, None]
    middle = centres[np.argmin(np.linalg.norm(centres - places.mean(axis=0), axis=1))]
    return tuple(float(value) for value in middle - (even.side - 1) / 2.0)


def _pull_profile(cells_per_side, boxes_per_cell_side, blob, cone_degrees, settings, device) -> dict:
    """Mean cell potential per distance bin, for cells near a face axis and near a body diagonal."""
    universe = blob_start(cells_per_side, boxes_per_cell_side, [dict(blob, velocity=(0.0, 0.0, 0.0))], device=device)
    ball = (universe.side - 1) / 2.0 + np.asarray(blob["offset"], dtype=np.float64)
    step(universe, settings)
    stats(universe, settings)
    boxes = universe.cell_boxes.numpy().astype(np.float64)
    living = (universe.alive.numpy() == 1) & (boxes > 0)
    places = universe.cell_place.numpy().reshape(-1, 3).astype(np.float64)[living] / boxes[living, None]
    potential = universe.cell_potential_mean.numpy().astype(np.float64)[living]
    offsets = places - ball
    distance = np.linalg.norm(offsets, axis=1)
    unit = offsets / np.maximum(distance, 1e-12)[:, None]
    nearest_axis = np.degrees(np.arccos(np.clip(np.abs(unit).max(axis=1), -1.0, 1.0)))
    nearest_diagonal = np.degrees(np.arccos(np.clip(np.abs(unit).sum(axis=1) / math.sqrt(3.0), -1.0, 1.0)))
    axis_cells, diagonal_cells = nearest_axis <= cone_degrees, nearest_diagonal <= cone_degrees
    width = float(boxes_per_cell_side)
    farthest = min(distance[axis_cells].max(initial=0.0), distance[diagonal_cells].max(initial=0.0))
    edges = np.arange(blob["radius"] + 1.0, farthest + width, width)
    profile = {"distance": [], "axis": [], "diagonal": [], "axis_cells": [], "diagonal_cells": [],
               "relative_difference": []}
    for inner, outer in zip(edges[:-1], edges[1:]):
        in_bin = (distance >= inner) & (distance < outer)
        on_axis, on_diagonal = in_bin & axis_cells, in_bin & diagonal_cells
        if not on_axis.any() or not on_diagonal.any():
            continue
        axis_mean, diagonal_mean = float(potential[on_axis].mean()), float(potential[on_diagonal].mean())
        profile["distance"].append(float(0.5 * (inner + outer)))
        profile["axis"].append(axis_mean)
        profile["diagonal"].append(diagonal_mean)
        profile["axis_cells"].append(int(on_axis.sum()))
        profile["diagonal_cells"].append(int(on_diagonal.sum()))
        profile["relative_difference"].append((diagonal_mean - axis_mean) / abs(axis_mean) if axis_mean else 0.0)
    profile["mean_relative_difference"] = float(np.mean(np.abs(profile["relative_difference"]))) \
        if profile["relative_difference"] else 0.0
    return profile


def _shift(cells_per_side, boxes_per_cell_side, blob, velocity, steps, settings, device) -> tuple:
    """How far the centre of all the energy moved over the run, and the start's momentum over its total energy."""
    universe = blob_start(cells_per_side, boxes_per_cell_side, [dict(blob, velocity=velocity)], device=device)
    owner = universe.owner.numpy()
    field = wave_field(universe)
    energy = np.abs(field) ** 2
    start = energy_centre(energy)
    momentum = _lattice_momentum(field, owner) / float(energy.sum())
    for _ in range(steps):
        step(universe, settings)
    return energy_centre(np.abs(wave_field(universe)) ** 2) - start, momentum


def _lattice_momentum(field: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Sum over box faces inside one cell of Im(conj psi_a psi_b), per axis: the flow the hops make."""
    momentum = np.zeros(3)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        same = (owner[tuple(lower)] >= 0) & (owner[tuple(lower)] == owner[tuple(upper)])
        momentum[axis] = float(np.imag(np.conj(field[tuple(lower)]) * field[tuple(upper)])[same].sum())
    return momentum


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
