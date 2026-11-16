"""Isotropy scene: is the foam the same in every direction? Pull and flow along a grid axis against a diagonal.

Contract: `isotropy(cells_per_side=12, boxes_per_cell_side=4, blob_energy=500.0,
blob_radius=3.0, flow_radius=16.0, speed=0.3, steps=20, seeds=(1, 2, 3, 4),
cone_degrees=20.0, pull_steps=200, settings=None, device="cuda")` makes two measurements on
the even checkerboard (default settings Settings(), every rule on). The Planck boxes form a cubic grid
with six face directions, so a direction-free foam should show no difference between a
face axis and a diagonal. Every ball sits on the middle energy volume: the one whose
box centre lies nearest the centre of all the energy volumes' boxes (with an even
cells_per_side the energy volumes, at all-odd cell places, are not centred on the grid),
so the foam's own pull and push are even round it.
Pull: one ball of extra energy blob_energy (units of 1000 quanta, blob_start.py) and
radius blob_radius at rest, `pull_steps` steps (the pull is a wave from light speed,
physics/engine/pull_wave.py: 200 steps of 0.25 let it cross 50 boxes, so it has formed
round the ball; the ball's own fluid moves meanwhile), then every occupied box's pull
against its distance from the ball. A box counts as "axis" when its direction from the
ball lies within cone_degrees of a face axis (plus or minus x, y, z), "diagonal" when
within cone_degrees of a body diagonal (plus or minus 1, 1, 1). The pull is 0 in empty
boxes (space ends there), so the filled cube's faces bound it: nearer along an axis than
along a diagonal.
Distances are binned one cell side wide from blob_radius + 1 out to the farthest bin
both sets reach.
Flow: a push big enough to measure (round 3, 2026-11-16: the earlier ball of radius 3 and
density 1 moved the energy centre 0.004 boxes, far under the engine's run-to-run scatter).
The foam's own quanta (1000 per box, nothing added) are given flow only, no phase jump: phase
= speed (x - centre) . direction, times min(1, flow_radius / |x - centre|), so the
speed is `speed` inside flow_radius and the phase levels off outside it (known shift
about 2.35 boxes at the defaults). Along x, the face diagonal (1, 1, 0) / sqrt 2 and the
body diagonal (1, 1, 1) / sqrt 3, `steps` steps each, and once at rest, for each seed in
`seeds` (each seed is the whole-number rounding's seed, so several seeds average its
scatter). What
is followed is the centre of all the whole quanta (energy_centre.py over every box, plus
and minus), so energy that crosses into joints, as sound or into joints,
still counts; each seed's rest run (the foam's own pull and push) is taken off its
moving runs. Its known value is where the start's momentum carries it: the start's
lattice momentum P = sum over box faces between two boxes of one sign of Im(conj psi_a
psi_b) along the face (the flow the hops make at openness 1), over the total quanta,
times the time; hops pass nothing across a plus-minus face, and the leak across a
joint (physics.engine.cross) turns or stops some of it with friction, which is
what "carried" shows.
Returns a dict (no verdict; same in every direction means the ratios are 1 and the
differences 0):
  cells_per_side, boxes_per_cell_side, steps, seeds, settings (dict), blob_energy,
  flow_radius, speed, ball_offset (the ball's place from the grid's middle, boxes)
  pull     distance (bin middles, boxes), axis, diagonal (mean box pull per bin),
           axis_cells, diagonal_cells (boxes per bin), relative_difference per bin
           ((diagonal - axis) / |axis|), mean_relative_difference (mean of its absolute)
  flow     per direction name ("axis", "face_diagonal", "body_diagonal"): along,
           across (boxes, mean over seeds of the energy centre's shift less that seed's
           rest run's), known (the start momentum's shift along the direction), carried
           (along / known), carried_sd (its spread over seeds, 0 for one seed);
           face_ratio and body_ratio (that diagonal's carried over the axis's);
           rest_shift (the rest runs' mean shift, 3 floats); rest_scatter (the farthest
           a rest run's shift lies from that mean, boxes: a moving run's along or across
           smaller than this is not measured)
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.engine import Settings, State, cells, host_boxes, lattice_of, new_universe, step
from physics.scenes.blob_start import blob_start
from physics.scenes.energy_centre import energy_centre
from physics.scenes.wave_field import wave_field

DIRECTIONS = {"axis": (1.0, 0.0, 0.0), "face_diagonal": (1.0, 1.0, 0.0), "body_diagonal": (1.0, 1.0, 1.0)}


def isotropy(cells_per_side: int = 12, boxes_per_cell_side: int = 4, blob_energy: float = 500.0,
             blob_radius: float = 3.0, flow_radius: float = 16.0, speed: float = 0.3, steps: int = 20,
             seeds: tuple = (1, 2, 3, 4), cone_degrees: float = 20.0, pull_steps: int = 200,
             settings: Settings | None = None, device: str = "cuda") -> dict:
    """Pull profile along axes against diagonals, and how far a pushed foam's energy goes along each."""
    settings = settings or Settings()
    offset = _middle_volume_offset(cells_per_side, boxes_per_cell_side, device)
    pull_blob = {"offset": offset, "radius": blob_radius, "energy": blob_energy}
    push = {"offset": offset, "radius": flow_radius, "speed": speed}
    return {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "steps": steps,
            "seeds": list(seeds), "settings": asdict(settings), "blob_energy": blob_energy,
            "flow_radius": flow_radius, "speed": speed, "ball_offset": [float(value) for value in offset],
            "pull_steps": pull_steps,
            "pull": _pull_profile(cells_per_side, boxes_per_cell_side, pull_blob, cone_degrees, pull_steps,
                                  settings, device),
            "flow": _flow(cells_per_side, boxes_per_cell_side, push, steps, seeds, settings, device)}


def _flow(cells_per_side, boxes_per_cell_side, push, steps, seeds, settings, device) -> dict:
    """Carried share per direction, each seed's rest run taken off, averaged over the seeds."""
    rests, runs = [], {name: [] for name in DIRECTIONS}
    known = {}
    for seed in seeds:
        rest, _ = _shift(cells_per_side, boxes_per_cell_side, push, (0.0, 0.0, 0.0), seed, steps, settings, device)
        rests.append(rest)
        for name, direction in DIRECTIONS.items():
            unit = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
            shift, momentum = _shift(cells_per_side, boxes_per_cell_side, push, tuple(unit), seed, steps,
                                     settings, device)
            known[name] = float(momentum @ unit) * steps * settings.step_time
            runs[name].append((shift - rest, unit))
    flow = {}
    for name, moves in runs.items():
        along = np.array([float(moved @ unit) for moved, unit in moves])
        across = np.array([float(np.linalg.norm(moved - (moved @ unit) * unit)) for moved, unit in moves])
        spread = float(along.std(ddof=1)) if along.size > 1 else 0.0
        flow[name] = {"along": float(along.mean()), "across": float(across.mean()), "known": known[name],
                      "carried": _ratio(float(along.mean()), known[name]), "carried_sd": _ratio(spread, known[name])}
    flow["face_ratio"] = _ratio(flow["face_diagonal"]["carried"], flow["axis"]["carried"])
    flow["body_ratio"] = _ratio(flow["body_diagonal"]["carried"], flow["axis"]["carried"])
    rests = np.array(rests)
    flow["rest_shift"] = [float(value) for value in rests.mean(axis=0)]
    flow["rest_scatter"] = float(np.linalg.norm(rests - rests.mean(axis=0), axis=1).max())
    return flow


def _middle_volume_offset(cells_per_side, boxes_per_cell_side, device) -> tuple:
    """The box centre of the energy volume nearest all energy volumes' centre, from the grid's middle."""
    even = new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0.0, device=device)
    boxes = host_boxes(even)
    found = cells(boxes)
    centres = found["centre"][:found["volumes"]]
    places = np.argwhere(boxes["quanta"] > 0).astype(np.float64)
    middle = centres[np.argmin(np.linalg.norm(centres - places.mean(axis=0), axis=1))]
    return tuple(float(value) for value in middle - (even.side - 1) / 2.0)


def _pull_profile(cells_per_side, boxes_per_cell_side, blob, cone_degrees, pull_steps, settings, device) -> dict:
    """Mean box pull per distance bin, for boxes near a face axis and near a body diagonal."""
    universe = blob_start(cells_per_side, boxes_per_cell_side, [dict(blob, velocity=(0.0, 0.0, 0.0))], device=device)
    ball = (universe.side - 1) / 2.0 + np.asarray(blob["offset"], dtype=np.float64)
    for _ in range(pull_steps):
        step(universe, settings)
    boxes = host_boxes(universe)
    occupied = boxes["quanta"] != 0
    places = np.argwhere(occupied).astype(np.float64)
    potential = boxes["pull"][occupied].astype(np.float64)
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


def _shift(cells_per_side, boxes_per_cell_side, push, unit, seed, steps, settings, device) -> tuple:
    """How far the centre of all the energy moved over the run, and the start's momentum over its total energy."""
    universe = _pushed_start(cells_per_side, boxes_per_cell_side, push, unit, seed, device)
    quanta = host_boxes(universe)["quanta"]
    field = wave_field(universe)
    energy = np.abs(field) ** 2
    start = energy_centre(energy)
    kinds = np.where(quanta > 0, 1, np.where(quanta < 0, 2, -1))
    momentum = _lattice_momentum(field, kinds) / float(energy.sum())
    for _ in range(steps):
        step(universe, settings)
    return energy_centre(np.abs(wave_field(universe)) ** 2) - start, momentum


def _pushed_start(cells_per_side, boxes_per_cell_side, push, unit, seed, device):
    """The even foam given flow along `unit` at push["speed"] inside push["radius"], levelling off outside it."""
    even = new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0.0, device=device)
    quanta = host_boxes(even)["quanta"]
    side = even.side
    places = np.stack(np.meshgrid(*[np.arange(side, dtype=np.float64)] * 3, indexing="ij"), axis=-1)
    offsets = places - ((side - 1) / 2.0 + np.asarray(push["offset"], dtype=np.float64))
    distance = np.linalg.norm(offsets, axis=-1)
    taper = np.minimum(1.0, push["radius"] / np.maximum(distance, 1e-12))
    phase = push["speed"] * (offsets @ np.asarray(unit, dtype=np.float64)) * taper
    phase[quanta == 0] = 0.0
    return State.from_host(quanta, phase, seed=seed, device=device, lattice=lattice_of(even))


def _lattice_momentum(field: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Sum over box faces between two boxes of one kind of Im(conj psi_a psi_b), per axis: the flow the hops make."""
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
