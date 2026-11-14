"""Gravity test: with the attraction rule off, do two energy lumps approach?

Contract: `run(seed, out_dir)` builds a 3D spatial reference network, places
two lumps of extra energy (`add_lumps`) a fixed network distance apart and
steps the rules in three runs: attraction off (energy flow and energy-length
on, drift off), attraction on (same plus attraction), and a control with no
lumps. Each step it reads, from the network only, the distance between the two
starting joints, the lump centres, widths and peaks, and the network's overall
size. Distances are divided by that size so a global shrink from the extra
energy is not mistaken for an approach. A light-bending probe compares the
shortest paths between far joint pairs before and after one lump settles.
Writes PNGs to `out_dir`; returns the JSON-serialisable result dict.

The verdict rule is fixed here, before any run: the attraction-off lumps
"approach" when their size-relative centre distance falls by more than
APPROACH_THRESHOLD beyond the control's change; if their peak excess has
dropped below IDENTITY_FLOOR of its start, the lumps have dissolved and the
verdict is "unclear" whatever the distance did. An approach counts as a "pass"
only if the lump centres also travel toward each other by TRAVEL_THRESHOLD on a
fixed ruler (the undisturbed network); an approach with centres that stay on
the same joints is the gap between them shrinking as energy spreads into it,
and is "unclear". (This travel condition was added after the first run showed
exactly that case; it makes the rule stricter, never looser.)
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict

import numpy as np

from physics.experiments.gravity.bending import path_closest_approach
from physics.experiments.gravity.lumps import add_lumps
from physics.experiments.gravity.plots import plot_curves
from physics.experiments.gravity.tracking import track_lumps
from physics.measure import distances, mean_length
from physics.network import Network, spatial_reference
from physics.rules import RuleSettings, step

JOINTS = 12_000
DIMENSION = 3
STEPS = 60
SEPARATION = 10.0       # lump separation, in mean connection lengths
LUMP_RADIUS = 2.0       # lump radius, in mean connection lengths
AMPLITUDE = 4.0         # lump connections hold 1 + AMPLITUDE times their energy
ENERGY_FLOW = 0.1
ATTRACTION_PER_DENSITY = 0.01   # attraction shortening per step at background density
APPROACH_THRESHOLD = 0.03
TRAVEL_THRESHOLD = 0.5  # fixed-ruler centre movement (mean lengths) that counts as travel
IDENTITY_FLOOR = 0.25
BENDING_PAIRS = 24


def run(seed: int, out_dir: str) -> dict:
    """Run the gravity test, write its plots into out_dir and return the result."""
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    base, known = spatial_reference(JOINTS, DIMENSION, seed=seed)
    unit = mean_length(base)
    background = float(np.median(_joint_density(base)))
    joints = _pick_pair(base, rng, SEPARATION * unit)
    reference_sources = rng.choice(JOINTS, 8, replace=False)
    lumped = add_lumps(base, list(joints), LUMP_RADIUS * unit, AMPLITUDE)
    attraction = ATTRACTION_PER_DENSITY / background
    off = RuleSettings(energy_length=1.0, energy_flow=ENERGY_FLOW)
    on = RuleSettings(energy_length=1.0, energy_flow=ENERGY_FLOW, attraction=attraction)
    histories = {
        "attraction off": _history(lumped, off, joints, background, reference_sources),
        "attraction on": _history(lumped, on, joints, background, reference_sources),
        "control, no lumps": _history(base, off, joints, background, reference_sources),
    }
    ruler = distances(base, list(joints)) / unit
    bending = _bending(base, joints[0], off, rng, unit)
    plots = _plot_all(out_dir, histories, unit, ruler)
    settings = {"joints": JOINTS, "dimension": DIMENSION, "neighbours": 2 * DIMENSION + 4,
                "steps": STEPS, "separation_in_mean_lengths": SEPARATION,
                "lump_radius_in_mean_lengths": LUMP_RADIUS, "amplitude": AMPLITUDE,
                "attraction_off_rules": asdict(off), "attraction_on_rules": asdict(on),
                "attraction_per_background_density": ATTRACTION_PER_DENSITY,
                "approach_threshold": APPROACH_THRESHOLD, "identity_floor": IDENTITY_FLOOR,
                "bending_pairs": BENDING_PAIRS, "known_dimension": known["dimension"]}
    return _result(histories, bending, plots, settings, seed, time.time() - started, ruler)


def _joint_density(network: Network) -> np.ndarray:
    sums = np.bincount(network.pairs.ravel(), weights=np.repeat(network.energy, 2),
                       minlength=network.joint_count)
    return sums / np.maximum(network.degree(), 1)


def _pick_pair(network: Network, rng: np.random.Generator, separation: float) -> tuple[int, int]:
    first = int(rng.integers(network.joint_count))
    from_first = distances(network, [first])[0]
    return first, int(np.argmin(np.abs(from_first - separation)))


def _history(network: Network, settings: RuleSettings, joints: tuple[int, int],
             background: float, reference_sources: np.ndarray) -> list[dict]:
    """Track the lumps each step; stop early if the rules drive a length to zero (collapse)."""
    records = [track_lumps(network, joints, background, reference_sources)]
    for _ in range(STEPS):
        with np.errstate(all="ignore"):
            network = step(network, settings)
        if not (np.all(np.isfinite(network.length)) and np.all(network.length > 0)):
            break
        records.append(track_lumps(network, joints, background, reference_sources))
    return records


def _centre_shift(records: list[dict], ruler: np.ndarray) -> np.ndarray:
    """Fixed-ruler distance between the two lump centres (undisturbed network, mean lengths).

    The ruler never changes, so this moves only if a centre moves to other joints:
    a lump travelling through the network, not the connections between it shrinking.
    """
    return np.array([ruler[0, record["lumps"][1]["centre"]] + ruler[1, record["lumps"][0]["centre"]]
                     for record in records]) / 2


def _relative(records: list[dict], key: str) -> np.ndarray:
    values = np.array([record[key] / record["scale"] for record in records])
    return values / values[0]


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


def _plot_all(out_dir: str, histories: dict, unit: float, ruler: np.ndarray) -> list[str]:
    names = ["joint_distance_relative.png", "centre_distance_relative.png",
             "joint_distance_raw.png", "lump_identity.png", "centre_fixed_ruler.png"]
    plot_curves(os.path.join(out_dir, names[0]),
                "Distance between the two starting joints / network size (1 = start)",
                "step", "relative distance",
                {label: (np.arange(len(records)), _relative(records, "joint_distance"))
                 for label, records in histories.items()},
                known=1.0, known_label="no motion (control expectation)")
    plot_curves(os.path.join(out_dir, names[1]),
                "Distance between lump centres / network size (1 = start)",
                "step", "relative centre distance",
                {label: (np.arange(len(records)), _relative(records, "centre_distance"))
                 for label, records in histories.items() if "control" not in label},
                known=1.0, known_label="no motion")
    plot_curves(os.path.join(out_dir, names[2]),
                "Raw network distance between the starting joints",
                "step", "distance (initial mean connection lengths)",
                {label: (np.arange(len(records)), np.array([record["joint_distance"] for record in records]) / unit)
                 for label, records in histories.items()},
                known=SEPARATION, known_label="placed separation")
    identity = {}
    for label, records in histories.items():
        if "control" in label:
            continue
        times = np.arange(len(records))
        peaks = np.array([record["lumps"][0]["peak"] for record in records])
        widths = np.array([record["lumps"][0]["width"] for record in records]) / unit
        identity[f"{label}: peak excess (share of start)"] = (times, peaks / peaks[0])
        identity[f"{label}: width (mean lengths / {LUMP_RADIUS:g})"] = (times, widths / LUMP_RADIUS)
    plot_curves(os.path.join(out_dir, names[3]),
                "Lump A identity: peak excess energy density and width",
                "step", "relative value (log scale)", identity,
                known=IDENTITY_FLOOR, known_label="identity floor (dissolved below)", log_y=True)
    plot_curves(os.path.join(out_dir, names[4]),
                "Lump centres on a fixed ruler: do the lumps travel through the network?",
                "step", "centre separation, undisturbed lengths (mean lengths)",
                {label: (np.arange(len(records)), _centre_shift(records, ruler))
                 for label, records in histories.items() if "control" not in label},
                known=SEPARATION, known_label="placed separation")
    return names


def _result(histories: dict, bending: dict, plots: list[str], settings: dict,
            seed: int, seconds: float, ruler: np.ndarray) -> dict:
    off, on, control = (histories[label] for label in
                        ("attraction off", "attraction on", "control, no lumps"))
    off_change = _relative(off, "centre_distance")[-1] - 1
    on_change = _relative(on, "centre_distance")[-1] - 1
    joint_off = _relative(off, "joint_distance")[-1] - 1
    control_change = _relative(control, "joint_distance")[-1] - 1
    on_steps = len(on) - 1
    peak_kept = off[-1]["lumps"][0]["peak"] / off[0]["lumps"][0]["peak"]
    shift_off = _centre_shift(off, ruler)
    shift_off = float(shift_off[-1] - shift_off[0])
    approach = off_change - control_change < -APPROACH_THRESHOLD
    dissolved = peak_kept < IDENTITY_FLOOR
    travelled = shift_off < -TRAVEL_THRESHOLD
    if dissolved or (approach and not travelled):
        verdict = "unclear"
    else:
        verdict = "pass" if approach else "fail"
    numbers = [
        {"name": "attraction off: change in size-relative lump-centre distance", "measured": off_change,
         "known": None, "note": f"approach needs < control - {APPROACH_THRESHOLD}"},
        {"name": "attraction off: change in size-relative starting-joint distance", "measured": joint_off,
         "known": None, "note": "same, but the fixed starting joints instead of tracked centres"},
        {"name": "control (no lumps): change in size-relative joint distance", "measured": control_change,
         "known": 0.0, "note": "nothing should move without lumps"},
        {"name": "attraction off: lump-centre separation change on a fixed ruler (mean lengths)",
         "measured": shift_off, "known": None,
         "note": "0 means the centres stayed on the same joints: the gap shrank, nothing travelled"},
        {"name": "attraction on: change in size-relative lump-centre distance", "measured": on_change,
         "known": None, "note": f"attraction {settings['attraction_on_rules']['attraction']:.4g}; "
                                f"measured at step {on_steps} of {STEPS}"},
        {"name": "attraction on: steps before a connection length collapsed to zero",
         "measured": on_steps, "known": None,
         "note": f"{STEPS} means no collapse within the run"},
        {"name": "attraction off: lump A peak excess kept at the end (share of start)",
         "measured": peak_kept, "known": None,
         "note": f"below {IDENTITY_FLOOR} counts as dissolved"},
        {"name": "light bending: mean closest approach of paths to a lump, before",
         "measured": bending["mean_before"], "known": None,
         "note": f"{bending['pairs']} far joint pairs, fixed flat ruler, mean lengths"},
        {"name": "light bending: mean closest approach, after the lump settles",
         "measured": bending["mean_after"], "known": None,
         "note": f"{bending['closer']} paths moved closer, {bending['farther']} farther; the "
                 "lump's short connections act as a shortcut, whereas in general relativity "
                 "space near a mass is stretched, so the sign matches but the mechanism differs"},
    ]
    return {"milestone": 3, "title": "Gravity test",
            "question": "With the attraction rule off, do two lumps of energy move toward "
                        "each other from the network's shape alone?",
            "verdict": verdict, "summary": _summary(verdict, off_change, control_change,
                                                    shift_off, on_steps, peak_kept, bending),
            "numbers": numbers, "plots": plots, "settings": settings, "seed": seed,
            "seconds": round(seconds, 1)}


def _summary(verdict: str, off_change: float, control_change: float, shift_off: float,
             on_steps: int, peak_kept: float, bending: dict) -> str:
    motion = "did not get closer" if verdict == "fail" else "got closer"
    travel = ("the lumps themselves stayed put and the gap between them shrank as their energy "
              "spread into it" if abs(shift_off) < TRAVEL_THRESHOLD else "the lump centres moved")
    return (f"Two lumps of extra energy were placed in a 3D network with the attraction rule "
            f"switched off; they {motion} (distance change {off_change:+.1%} against "
            f"{control_change:+.1%} in an empty control). Lump A kept {peak_kept:.0%} of its "
            f"peak extra energy, and its centre moved {shift_off:+.2f} connection lengths "
            f"on a fixed ruler, so {travel}. "
            f"With attraction on, the network collapsed after {on_steps} of {STEPS} steps. "
            f"Shortest paths passing a lump moved on average {bending['mean_shift']:+.2f} "
            f"connection lengths ({bending['closer']} of {bending['pairs']} closer).")
