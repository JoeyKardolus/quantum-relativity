"""Gravity test: do two energy lumps approach, and does the v2 joint push hold a lump up?

Contract: `run(seed, out_dir)` builds a 3D spatial reference network, places
two lumps of extra energy (`add_lumps`) a fixed network distance apart and
steps the rules in three runs: attraction off (energy flow and energy-length
on, drift off), attraction on (same plus the v2 pull, joint push and a finite
push limit), and a control with no lumps. Each step it reads, from the network
only, the distance between the two starting joints, the lump centres, widths
and peaks, and the network's overall size (`lump_history`). A fourth set of
runs puts one heavy lump past the push limit (`heavy_lump_run`) to see whether
joint collapse runs away. A light-bending probe compares shortest paths before
and after one lump settles. Writes PNGs to `out_dir` (picture: the network
itself before and after the attraction-on run); returns the result dict
(`gravity_result`).

Settings fixed before any v2 run: attraction makes the pull shorten a
background connection by ATTRACTION_PER_PULL per step; a joint pushes back
JOINT_PUSH of its squeeze; the push limit is LIMIT_MULTIPLE times the push a
background joint gives, so the two ordinary lumps start under it and the
heavy lumps start past it.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict

import numpy as np

from physics.experiments.gravity.bending import path_closest_approach
from physics.experiments.gravity.heavy import heavy_lump_run
from physics.experiments.gravity.history import lump_history
from physics.experiments.gravity.lumps import add_lumps
from physics.experiments.gravity.curves import lump_curves
from physics.experiments.gravity.plots import plot_curves
from physics.experiments.gravity.result import HELD_SHARE, IDENTITY_FLOOR, gravity_result
from physics.experiments.picture import network_picture
from physics.measure import distances, mean_length
from physics.network import Network, spatial_reference
from physics.rules import RuleSettings, pull, step

JOINTS = 6_000
DIMENSION = 3
STEPS = 60
SEPARATION = 8.0        # lump separation, in mean connection lengths
LUMP_RADIUS = 2.0       # lump radius, in mean connection lengths
AMPLITUDE = 4.0         # lump connections hold 1 + AMPLITUDE times their energy
HEAVY_AMPLITUDES = (8.0, 12.0, 20.0)
ENERGY_FLOW = 0.1
ATTRACTION_PER_PULL = 0.01   # log-shortening per step at the background pull
JOINT_PUSH = 0.5
LIMIT_MULTIPLE = 2.0
PICTURE_MARGIN = 4.0    # picture: joints with d(A) + d(B) <= SEPARATION + margin (mean lengths)
MIN_DRAWN = 20          # picture: below this many region joints, draw the whole network
BENDING_PAIRS = 24


def run(seed: int, out_dir: str) -> dict:
    """Run the gravity test, write its plots into out_dir and return the result."""
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    base, known = spatial_reference(JOINTS, DIMENSION, seed=seed)
    unit = mean_length(base)
    background = float(np.median(_joint_density(base)))
    background_pull = float(np.median(pull(base, base.energy, base.length)))
    joints = _pick_pair(base, rng, SEPARATION * unit)
    reference_sources = rng.choice(JOINTS, 8, replace=False)
    lumped = add_lumps(base, list(joints), LUMP_RADIUS * unit, AMPLITUDE)
    attraction = ATTRACTION_PER_PULL / background_pull
    off = RuleSettings(energy_length=1.0, energy_flow=ENERGY_FLOW)
    on = RuleSettings(energy_length=1.0, energy_flow=ENERGY_FLOW, attraction=attraction,
                      joint_push=JOINT_PUSH,
                      push_limit=LIMIT_MULTIPLE * JOINT_PUSH * ATTRACTION_PER_PULL)
    runs = {"attraction off": lump_history(lumped, off, joints, background, reference_sources, STEPS),
            "attraction on": lump_history(lumped, on, joints, background, reference_sources, STEPS),
            "control, no lumps": lump_history(base, off, joints, background, reference_sources, STEPS)}
    heavy = [heavy_lump_run(base, joints[0], LUMP_RADIUS * unit, amplitude, on, STEPS)
             for amplitude in HEAVY_AMPLITUDES]
    ruler = distances(base, list(joints)) / unit
    bending = _bending(base, joints[0], off, rng, unit)
    histories = {label: records for label, (records, _, _) in runs.items()}
    plots = _plot_all(out_dir, histories, heavy, unit, ruler)
    picture = _picture(out_dir, runs["attraction on"], ruler)
    settings = {"joints": JOINTS, "dimension": DIMENSION, "neighbours": 2 * DIMENSION + 4,
                "steps": STEPS, "separation_in_mean_lengths": SEPARATION,
                "lump_radius_in_mean_lengths": LUMP_RADIUS, "amplitude": AMPLITUDE,
                "heavy_amplitudes": list(HEAVY_AMPLITUDES),
                "attraction_off_rules": asdict(off), "attraction_on_rules": asdict(on),
                "attraction_per_background_pull": ATTRACTION_PER_PULL,
                "background_pull": background_pull, "limit_multiple": LIMIT_MULTIPLE,
                "lump_pull_ratio": float(pull(lumped, lumped.energy, lumped.length)[joints[0]]
                                         / background_pull),
                "bending_pairs": BENDING_PAIRS, "known_dimension": known["dimension"]}
    stopped = {label: note for label, (_, _, note) in runs.items()}
    return gravity_result(histories, stopped, heavy, bending, plots, picture, settings, seed,
                          time.time() - started, ruler)


def _joint_density(network: Network) -> np.ndarray:
    sums = np.bincount(network.pairs.ravel(), weights=np.repeat(network.energy, 2),
                       minlength=network.joint_count)
    return sums / np.maximum(network.degree(), 1)


def _pick_pair(network: Network, rng: np.random.Generator, separation: float) -> tuple[int, int]:
    first = int(rng.integers(network.joint_count))
    from_first = distances(network, [first])[0]
    return first, int(np.argmin(np.abs(from_first - separation)))


def _bending(base: Network, lump_joint: int, settings: RuleSettings,
             rng: np.random.Generator, unit: float) -> dict:
    """Shortest paths passing 2-4 mean lengths from one settled lump: do they move closer?"""
    one_lump = add_lumps(base, [lump_joint], LUMP_RADIUS * unit, AMPLITUDE)
    for _ in range(5):
        one_lump = step(one_lump, settings)
    ruler = distances(base, [lump_joint])[0] / unit
    ring = np.flatnonzero((ruler > 6) & (ruler < 9))
    before, after = [], []
    for _ in range(BENDING_PAIRS * 20):
        start, end = (int(joint) for joint in rng.choice(ring, 2, replace=False))
        flat = path_closest_approach(base, start, end, ruler)
        if 2.0 <= flat <= 4.0:
            before.append(flat)
            after.append(path_closest_approach(one_lump, start, end, ruler))
        if len(before) == BENDING_PAIRS:
            break
    shift = np.array(after) - np.array(before)
    return {"pairs": len(before), "mean_before": float(np.mean(before)),
            "mean_after": float(np.mean(after)), "mean_shift": float(np.mean(shift)),
            "closer": int(np.sum(shift < 0)), "farther": int(np.sum(shift > 0))}


def _picture(out_dir: str, on_run: tuple, ruler: np.ndarray) -> str:
    """The network around both lumps: at the start, at the last step before more than
    HELD_SHARE of the joints had collapsed, and at the last readable step of the attraction-on run."""
    _, snapshots, _ = on_run
    region = np.flatnonzero(ruler.sum(axis=0) <= SEPARATION + PICTURE_MARGIN)
    start_count = snapshots[0][0].joint_count
    held = max(index for index, (network, _) in enumerate(snapshots)
               if network.joint_count >= HELD_SHARE * start_count)
    chosen = sorted({0, held, len(snapshots) - 1})
    panels, carried = [], []
    for index in chosen:
        network, mapping = snapshots[index]
        joints = np.unique(mapping[region])
        if len(joints) < MIN_DRAWN:
            joints = np.arange(network.joint_count)
        panels.append((f"step {index}: {network.joint_count} joints", network, joints))
        carried.append(mapping[region])
    name = "network_before_after.png"
    network_picture(os.path.join(out_dir, name),
                    "Two energy lumps with pull and joint push on (2D layout for drawing only)",
                    panels, carried[1:])
    return name


def _plot_all(out_dir: str, histories: dict, heavy: list[dict], unit: float,
              ruler: np.ndarray) -> list[str]:
    names = ["joint_distance_relative.png", "centre_distance_relative.png",
             "joint_distance_raw.png", "lump_identity.png", "centre_fixed_ruler.png",
             "heavy_lump_joints.png"]
    curves = {label: lump_curves(records, ruler) for label, records in histories.items()}
    lumped = [label for label in histories if "control" not in label]
    plot_curves(os.path.join(out_dir, names[0]),
                "Distance between the two starting joints / network size (1 = start)",
                "step", "relative distance",
                {label: (np.arange(len(curve["joint"])), curve["joint"]) for label, curve in curves.items()},
                known=1.0, known_label="no motion (control expectation)")
    plot_curves(os.path.join(out_dir, names[1]),
                "Distance between lump centres / network size (1 = start)",
                "step", "relative centre distance",
                {label: (np.arange(len(curves[label]["centre"])), curves[label]["centre"]) for label in lumped},
                known=1.0, known_label="no motion")
    plot_curves(os.path.join(out_dir, names[2]),
                "Raw network distance between the starting joints",
                "step", "distance (initial mean connection lengths)",
                {label: (np.arange(len(records)), np.array([record["joint_distance"] for record in records]) / unit)
                 for label, records in histories.items()},
                known=SEPARATION, known_label="placed separation")
    identity = {}
    for label in lumped:
        times = np.arange(len(histories[label]))
        peaks = np.array([record["lumps"][0]["peak"] for record in histories[label]])
        widths = np.array([record["lumps"][0]["width"] for record in histories[label]]) / unit
        identity[f"{label}: peak excess (share of start)"] = (times, peaks / peaks[0])
        identity[f"{label}: width (mean lengths / {LUMP_RADIUS:g})"] = (times, widths / LUMP_RADIUS)
    plot_curves(os.path.join(out_dir, names[3]),
                "Lump A identity: peak excess energy density and width",
                "step", "relative value (log scale)", identity,
                known=IDENTITY_FLOOR, known_label="identity floor (dissolved below)", log_y=True)
    plot_curves(os.path.join(out_dir, names[4]),
                "Lump centres on a fixed ruler: do the lumps travel through the network?",
                "step", "centre separation, undisturbed lengths (mean lengths)",
                {label: (np.arange(len(curves[label]["fixed"])), curves[label]["fixed"]) for label in lumped},
                known=SEPARATION, known_label="placed separation")
    plot_curves(os.path.join(out_dir, names[5]),
                "One heavy lump past the push limit: joints left after joint collapse",
                "step", "joints", {f"lump amplitude {run['amplitude']:g}":
                                   (np.arange(len(run["joints"])), np.array(run["joints"]))
                                   for run in heavy},
                known=JOINTS, known_label="joints at start", log_y=True)
    return names
