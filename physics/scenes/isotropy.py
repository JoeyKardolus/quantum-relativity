"""Isotropy scene: is the foam the same in every direction? Pull and flow along a grid axis against a diagonal.

Contract: `isotropy(cells_per_side=12, boxes_per_cell_side=4, blob_energy=500.0,
blob_radius=3.0, speed=0.3, steps=20, cone_degrees=20.0, settings=None, device="cuda")`
makes two measurements on blob_start's even checkerboard (default settings
Settings.all_on()). The Planck boxes form a cubic grid with six face directions, so a
direction-free foam should show no difference between a face axis and a diagonal.
Pull: one ball of extra energy blob_energy and radius blob_radius at rest in the middle,
one step (the pull is solved in it), then every living cell's mean pull potential
(stats refreshes the cell sums) against its centre's distance from the ball. A cell
counts as "axis" when its direction from the middle lies within cone_degrees of a face
axis (plus or minus x, y, z), "diagonal" when within cone_degrees of a body diagonal
(plus or minus 1, 1, 1). Distances are binned one cell side wide from blob_radius + 1
out to the farthest bin both sets reach.
Flow: three runs of the same ball given `speed` (boxes per time) along x, along the face
diagonal (1, 1, 0) / sqrt 2 and along the body diagonal (1, 1, 1) / sqrt 3, `steps`
steps each. The ball is the A-energy above the even foam's 1 per A box (clipped at 0);
its centre (energy_centre.py) after the steps against where it started gives how far it
went along its direction and across it.
Returns a dict (no verdict; same in every direction means the ratios are 1 and the
differences 0):
  cells_per_side, boxes_per_cell_side, steps, settings (dict), blob_energy, speed
  pull     distance (bin middles, boxes), axis, diagonal (mean cell potential per bin),
           axis_cells, diagonal_cells (cells per bin), relative_difference per bin
           ((diagonal - axis) / |axis|), mean_relative_difference (mean of its absolute)
  flow     per direction name ("axis", "face_diagonal", "body_diagonal"): along,
           across (boxes), carried (along / (speed * time)); face_ratio and body_ratio
           (that diagonal's along over the axis's along)
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.potts import Settings, stats, step
from physics.scenes.blob_start import blob_start
from physics.scenes.box_energies import box_energies
from physics.scenes.energy_centre import energy_centre

FOAM_A = 1.0
DIRECTIONS = {"axis": (1.0, 0.0, 0.0), "face_diagonal": (1.0, 1.0, 0.0), "body_diagonal": (1.0, 1.0, 1.0)}


def isotropy(cells_per_side: int = 12, boxes_per_cell_side: int = 4, blob_energy: float = 500.0,
             blob_radius: float = 3.0, speed: float = 0.3, steps: int = 20, cone_degrees: float = 20.0,
             settings: Settings | None = None, device: str = "cuda") -> dict:
    """Pull profile along axes against diagonals, and how far a moving ball goes along each."""
    settings = settings or Settings.all_on()
    blob = {"offset": (0.0, 0.0, 0.0), "radius": blob_radius, "energy": blob_energy}
    flow = {name: _carried(cells_per_side, boxes_per_cell_side, blob, speed, direction, steps, settings, device)
            for name, direction in DIRECTIONS.items()}
    flow["face_ratio"] = _ratio(flow["face_diagonal"]["along"], flow["axis"]["along"])
    flow["body_ratio"] = _ratio(flow["body_diagonal"]["along"], flow["axis"]["along"])
    return {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "steps": steps,
            "settings": asdict(settings), "blob_energy": blob_energy, "speed": speed,
            "pull": _pull_profile(cells_per_side, boxes_per_cell_side, blob, cone_degrees, settings, device),
            "flow": flow}


def _pull_profile(cells_per_side, boxes_per_cell_side, blob, cone_degrees, settings, device) -> dict:
    """Mean cell potential per distance bin, for cells near a face axis and near a body diagonal."""
    universe = blob_start(cells_per_side, boxes_per_cell_side, [dict(blob, velocity=(0.0, 0.0, 0.0))], device=device)
    step(universe, settings)
    stats(universe, settings)
    boxes = universe.cell_boxes.numpy().astype(np.float64)
    living = (universe.alive.numpy() == 1) & (boxes > 0)
    places = universe.cell_place.numpy().reshape(-1, 3).astype(np.float64)[living] / boxes[living, None]
    potential = universe.cell_potential_mean.numpy().astype(np.float64)[living]
    offsets = places - (universe.side - 1) / 2.0
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


def _carried(cells_per_side, boxes_per_cell_side, blob, speed, direction, steps, settings, device) -> dict:
    """How far the moving ball's excess energy centre went along its direction and across it."""
    unit = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
    universe = blob_start(cells_per_side, boxes_per_cell_side, [dict(blob, velocity=tuple(speed * unit))],
                          device=device)
    start = _excess_centre(universe)
    for _ in range(steps):
        step(universe, settings)
    moved = _excess_centre(universe) - start
    along = float(moved @ unit)
    time = steps * settings.step_time
    return {"along": along, "across": float(np.linalg.norm(moved - along * unit)),
            "carried": along / (speed * time) if speed * time else 0.0}


def _excess_centre(universe) -> np.ndarray:
    a_energy, _ = box_energies(universe)
    excess = np.where(universe.owner.numpy() >= 0, np.maximum(a_energy - FOAM_A, 0.0), 0.0)
    return energy_centre(excess)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
