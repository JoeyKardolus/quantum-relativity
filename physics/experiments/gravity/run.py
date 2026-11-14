"""Gravity test: with every rule on and a balancing drift, do two energy lumps approach?

Contract: `run(seed, out_dir)` builds a 3D spatial reference network and makes
its energy even (every connection at the mean length, energy by the
energy-length rule): the scene's background. It calibrates the balancing drift
on that even background (`balancing_drift`, the one calibration), places two
lumps of extra energy (`add_lumps`) a fixed network distance apart and steps
three runs with every rule on (`all_rules_on`): the main run with the
balancing drift, a control with the same rules and no lumps, and the v2 way
(drift off) for comparison. Each step it reads, from the network only, the
lump centres, their distance, the distance between the starting joints and the
far background's mean connection length (joints at least FAR mean lengths from
both lumps), so whole-space shrink or growth is not counted as approach
(`lump_history`). Heavy lumps past the push limit run with the main rules
(`heavy_lump_run`) to see whether joint collapse runs away. Writes PNGs to
`out_dir` (picture: the background-relative lump distance of the three runs);
returns the result dict (`gravity_result`).

Settings fixed before any run: attraction makes the pull shorten a background
connection by ATTRACTION_PER_PULL per step; every other strength is
`all_rules_on`'s.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, replace

import numpy as np

from physics.experiments.gravity.curves import lump_curves
from physics.experiments.gravity.heavy import heavy_lump_run
from physics.experiments.gravity.history import lump_history
from physics.experiments.gravity.lumps import add_lumps
from physics.experiments.gravity.plots import plot_curves
from physics.experiments.gravity.result import (CONTROL, DRIFT_OFF, IDENTITY_FLOOR, MAIN,
                                                gravity_result)
from physics.experiments.picture import network_picture
from physics.measure import distances, mean_length
from physics.network import Network, spatial_reference
from physics.rules import all_rules_on, balancing_drift, pull

JOINTS = 6_000
DIMENSION = 3
STEPS = 60
HEAVY_STEPS = 30
SEPARATION = 8.0        # lump separation, in mean connection lengths
LUMP_RADIUS = 2.0       # lump radius, in mean connection lengths
AMPLITUDE = 4.0         # lump connections hold 1 + AMPLITUDE times their energy
HEAVY_AMPLITUDES = (8.0, 20.0)
ATTRACTION_PER_PULL = 0.01   # log-shortening per step at the background pull
FAR = 5.0               # far background: joints at least this many mean lengths from both lumps
PICTURE_MARGIN = 4.0    # network picture: joints with d(A) + d(B) <= SEPARATION + margin
MIN_DRAWN = 20          # network picture: below this many region joints, draw the whole network


def run(seed: int, out_dir: str) -> dict:
    """Run the gravity test, write its plots into out_dir and return the result."""
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    scattered, known = spatial_reference(JOINTS, DIMENSION, seed=seed)
    base = Network.from_lengths(scattered.joint_count, scattered.pairs,
                                np.full(scattered.connection_count, scattered.length.mean()))
    unit = mean_length(base)
    background_pull = float(np.median(pull(base, base.energy, base.length)))
    off = all_rules_on(attraction=ATTRACTION_PER_PULL / background_pull)
    drift = balancing_drift(base, off, rng)
    main = replace(off, drift=drift)
    joints = _pick_pair(base, rng, SEPARATION * unit)
    ruler = distances(base, list(joints)) / unit
    far_joints = np.flatnonzero(ruler.min(axis=0) >= FAR)
    lumped = add_lumps(base, list(joints), LUMP_RADIUS * unit, AMPLITUDE)
    runs = {MAIN: lump_history(lumped, main, joints, far_joints, STEPS),
            CONTROL: lump_history(base, main, joints, far_joints, STEPS),
            DRIFT_OFF: lump_history(lumped, off, joints, far_joints, STEPS)}
    heavy = [heavy_lump_run(base, joints[0], LUMP_RADIUS * unit, amplitude, main, HEAVY_STEPS)
             for amplitude in HEAVY_AMPLITUDES]
    histories = {label: records for label, (records, _, _) in runs.items()}
    plots = _plot_all(out_dir, histories, heavy, unit, ruler)
    plots.append(_network_picture(out_dir, runs[MAIN], ruler))
    settings = {"joints": JOINTS, "dimension": DIMENSION, "neighbours": 2 * DIMENSION + 4,
                "background": "even: every connection at the reference network's mean length",
                "steps": STEPS, "heavy_steps": HEAVY_STEPS, "separation_in_mean_lengths": SEPARATION,
                "lump_radius_in_mean_lengths": LUMP_RADIUS, "amplitude": AMPLITUDE,
                "heavy_amplitudes": list(HEAVY_AMPLITUDES), "far_in_mean_lengths": FAR,
                "far_joints": int(len(far_joints)), "main_rules": asdict(main),
                "drift_off_rules": asdict(off), "balancing_drift": drift,
                "attraction_per_background_pull": ATTRACTION_PER_PULL,
                "background_pull": background_pull,
                "lump_pull_ratio": float(pull(lumped, lumped.energy, lumped.length)[joints[0]]
                                         / background_pull),
                "known_dimension": known["dimension"]}
    stopped = {label: note for label, (_, _, note) in runs.items()}
    return gravity_result(histories, stopped, heavy, plots, plots[0], settings, seed,
                          time.time() - started, ruler, unit)


def _pick_pair(network: Network, rng: np.random.Generator, separation: float) -> tuple[int, int]:
    first = int(rng.integers(network.joint_count))
    from_first = distances(network, [first])[0]
    return first, int(np.argmin(np.abs(from_first - separation)))


def _plot_all(out_dir: str, histories: dict, heavy: list[dict], unit: float,
              ruler: np.ndarray) -> list[str]:
    names = ["gap_relative_to_background.png", "gap_raw.png", "background_size.png",
             "centre_fixed_ruler.png", "lump_identity.png", "heavy_lump_joints.png"]
    curves = {label: lump_curves(records, ruler, unit) for label, records in histories.items()}
    lumped = [label for label in histories if label != CONTROL]
    steps = {label: np.arange(len(records)) for label, records in histories.items()}
    gap = {label: (steps[label], curves[label]["centre"]) for label in lumped}
    gap[CONTROL] = (steps[CONTROL], curves[CONTROL]["joint"])
    plot_curves(os.path.join(out_dir, names[0]),
                "Lump distance in far-background connection lengths (1 = start)",
                "step", "relative distance", gap, known=1.0, known_label="no approach")
    plot_curves(os.path.join(out_dir, names[1]), "Raw lump-centre distance",
                "step", "distance (starting mean connection lengths)",
                {label: (steps[label], curves[label]["raw"]) for label in lumped})
    plot_curves(os.path.join(out_dir, names[2]),
                "Far-background mean connection length (1 = start): the size of space",
                "step", "relative length",
                {label: (steps[label], curve["background"]) for label, curve in curves.items()},
                known=1.0, known_label="space keeps its size")
    plot_curves(os.path.join(out_dir, names[3]),
                "Lump centres on a fixed ruler: do the lumps travel through the network?",
                "step", "centre separation, undisturbed lengths (mean lengths)",
                {label: (steps[label], curves[label]["fixed"]) for label in lumped},
                known=SEPARATION, known_label="placed separation")
    identity = {}
    for label in lumped:
        peaks = np.array([record["lumps"][0]["peak"] for record in histories[label]])
        identity[f"{label}: lump A peak excess (share of start)"] = (steps[label], peaks / peaks[0])
    plot_curves(os.path.join(out_dir, names[4]), "Lump A identity: peak excess energy density",
                "step", "relative value (log scale)", identity,
                known=IDENTITY_FLOOR, known_label="identity floor (dissolved below)", log_y=True)
    plot_curves(os.path.join(out_dir, names[5]),
                "One heavy lump past the push limit, all rules on: joints left after joint collapse",
                "step", "joints", {f"lump amplitude {run['amplitude']:g}":
                                   (np.arange(len(run["joints"])), np.array(run["joints"]))
                                   for run in heavy},
                known=JOINTS, known_label="joints at start", log_y=True)
    return names


def _network_picture(out_dir: str, main_run: tuple, ruler: np.ndarray) -> str:
    """The network around both lumps at the start, halfway and at the end of the main run."""
    _, snapshots, _ = main_run
    region = np.flatnonzero(ruler.sum(axis=0) <= SEPARATION + PICTURE_MARGIN)
    chosen = sorted({0, (len(snapshots) - 1) // 2, len(snapshots) - 1})
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
                    "Two energy lumps, all rules on, balancing drift (2D layout for drawing only)",
                    panels, carried[1:])
    return name
