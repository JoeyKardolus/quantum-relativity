"""Space test (milestone 5): start with no geometry, run the rules, watch the dimension.

Contract: `run(seed, out_dir)` builds four starts of JOINTS joints: a random
network (no geometry), a 6D and a 4D spatial reference, and a 3D spatial
reference as a control (is 3D space kept?). Every start runs STEPS steps with
rewiring only (energy flow, the energy-length rule and REWIRING_ATTEMPTS
rewiring attempts per step; target degree = the start's own mean degree,
since rewiring keeps the connection count; target triangles 0, the core
default guess). The ALL_RULES starts also run ALL_RULES_STEPS steps with all
rules on (`all_rules_on`, owner 2026-11-14) plus the same rewiring: attraction
= BACKGROUND_SHORTENING / the start's median pull, drift = `balancing_drift`
on the same connections all at the start's mean length (the one calibration,
reported), charges 0 so repulsion has nothing to act on. Only the 3D control
runs all rules: the pull's exact sparse solve takes about 8 s (4D), 16 s
(random) and 37 s (6D) per step at JOINTS joints against 1.4 s on 3D, which
does not fit the ~3 minute budget; those starts are reported as rewiring only.
The 3D control crumples within 10 steps under rewiring alone, so
ALL_RULES_STEPS (a checkpoint) is long enough to see whether the other rules
change that (33 steps took 5 minutes, the pull's solve slowing as the
network crumples). Rewiring reads only which joints connect, so the other
rules can change its outcome only through joint collapse. Dimension is measured at CHECKPOINTS, log-spaced (`track`).
Plots: out_dir holds the 3D control under both rule sets (picture: steps
across), out_dir/rewiring_only every start under rewiring only. Verdict, fixed
in advance, from the all-rules runs: pass if some start ends with both
dimensions within 10% of 3, unclear if one of them does, fail otherwise.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np

from physics.experiments.space.high_dimensional import high_dimensional_reference
from physics.experiments.space.plots import plot_tracks
from physics.experiments.space.track import track
from physics.network import Network, random_network, spatial_reference
from physics.rules import RuleSettings, all_rules_on, balancing_drift, pull

JOINTS = 10_000
STEPS = 120
REWIRING_ATTEMPTS = 5_000
CHECKPOINTS = sorted({0, *np.unique(np.geomspace(1, STEPS, 16).round().astype(int)).tolist()})
ALL_RULES = ("3D reference (control)",)
ALL_RULES_STEPS = 13
ENERGY_FLOW = 0.1
BACKGROUND_SHORTENING = 0.01
TARGET = 3.0
BAND = 0.10


def run(seed: int, out_dir: str) -> dict:
    """Run the space test and return its JSON-serialisable result."""
    began = time.time()
    starts = _starts(seed)
    tracks, alone, used = {}, {}, {}
    for name, (network, _known) in starts.items():
        rewiring_only = RuleSettings(energy_length=1.0, energy_flow=ENERGY_FLOW,
                                     rewiring=REWIRING_ATTEMPTS,
                                     target_degree=float(round(network.degree().mean())))
        used[name] = {"target_degree": rewiring_only.target_degree,
                      "connections": int(network.connection_count)}
        alone[name] = track(network, rewiring_only, STEPS, CHECKPOINTS, seed)
        if name in ALL_RULES:
            settings = _all_rules(network, rewiring_only, seed)
            used[name] |= {"attraction": settings.attraction,
                           "balancing_drift (calibration)": settings.drift}
            tracks[name] = track(network, settings, ALL_RULES_STEPS,
                                 [at for at in CHECKPOINTS if at <= ALL_RULES_STEPS], seed)
    side_by_side = {f"{name}, {label}": history for name in tracks
                    for label, history in (("all rules on", tracks[name]),
                                           ("rewiring only", alone[name]))}
    plots = plot_tracks(side_by_side, out_dir)
    plots += [os.path.join("rewiring_only", name)
              for name in plot_tracks(alone, os.path.join(out_dir, "rewiring_only"))]
    numbers = _numbers(starts, alone) + _comparison(tracks, alone)
    verdict = _verdict(tracks)
    return {
        "milestone": 5,
        "title": "Space test",
        "question": "Starting from a network with no geometry (or the wrong dimension), "
                    "do the rules with rewiring settle it into something that behaves "
                    "like 3D space?",
        "verdict": verdict,
        "summary": "With rewiring only, all four starts: " + _summary(_verdict(alone), alone),
        "meaning": _meaning(tracks, alone),
        "numbers": numbers,
        "plots": plots,
        "picture": "space_steps_across.png",
        "settings": {"joints": JOINTS, "steps": STEPS, "checkpoints": CHECKPOINTS,
                     "all_rules_starts": ALL_RULES, "all_rules_steps": ALL_RULES_STEPS,
                     "rewiring_attempts_per_step": REWIRING_ATTEMPTS,
                     "rules": "all_rules_on plus rewiring (3D control only; pull too slow on "
                              "the others at this size); attraction = "
                              f"{BACKGROUND_SHORTENING} / median pull; drift = balancing_drift; "
                              "charges 0",
                     "comparison": "rewiring only: energy flow 0.1, energy-length 1",
                     "target_triangles": 0.0, "triangle_weight": 1.0, "per_start": used,
                     "measures": "ball_dimension and spectral_dimension, default windows, "
                                 f"seed {seed}"},
        "seed": seed,
        "seconds": round(time.time() - began, 1),
    }


def _all_rules(network: Network, rewiring_only: RuleSettings, seed: int) -> RuleSettings:
    even = Network.from_lengths(network.joint_count, network.pairs,
                                np.full(network.connection_count, network.length.mean()))
    background = float(np.median(pull(even, even.energy, even.length)))
    settings = all_rules_on(attraction=BACKGROUND_SHORTENING / background,
                            target_degree=rewiring_only.target_degree)
    drift = balancing_drift(even, settings, np.random.default_rng(seed))
    return replace(settings, drift=drift, rewiring=rewiring_only.rewiring)


def _comparison(tracks: dict, alone: dict) -> list[dict]:
    rows = []
    for name, history in tracks.items():
        at = alone[name]["step"].index(history["step"][-1])
        rows += [
            {"name": f"{name}, step {history['step'][-1]}: ball / spectral dimension, all rules on vs "
                     "rewiring only",
             "measured": [[_round(history["ball"][-1]), _round(history["spectral"][-1])],
                          [_round(alone[name]["ball"][at]), _round(alone[name]["spectral"][at])]],
             "known": [TARGET, TARGET], "note": "None = no valid fit range (small-world)"},
            {"name": f"{name}, step {history['step'][-1]}: steps across, all rules on vs rewiring only",
             "measured": [round(history["across"][-1], 2), round(alone[name]["across"][at], 2)],
             "known": None, "note": f"joints left with all rules on: {history['joints'][-1]} of "
                                    f"{history['joints'][0]}"},
        ]
    return rows


def _starts(seed: int) -> dict:
    return {
        "random (no geometry)": (random_network(JOINTS, mean_degree=6.0, seed=seed), None),
        "6D reference": (high_dimensional_reference(JOINTS, 6, neighbours=16, seed=seed), 6),
        "4D reference": (spatial_reference(JOINTS, 4, seed=seed)[0], 4),
        "3D reference (control)": (spatial_reference(JOINTS, 3, seed=seed)[0], 3),
    }


def _near(value: float | None) -> bool:
    return value is not None and abs(value - TARGET) <= BAND * TARGET


def _verdict(tracks: dict) -> str:
    finals = [(history["ball"][-1], history["spectral"][-1]) for history in tracks.values()]
    if any(_near(ball) and _near(spectral) for ball, spectral in finals):
        return "pass"
    if any(_near(ball) or _near(spectral) for ball, spectral in finals):
        return "unclear"
    return "fail"


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _numbers(starts: dict, tracks: dict) -> list[dict]:
    rows = []
    for name, history in tracks.items():
        known = starts[name][1]
        undefined = "None = no valid fit range (small-world or too few joints)"
        rows += [
            {"name": f"{name}: ball dimension at step 0", "measured": _round(history["ball"][0]),
             "known": known, "note": f"start check of the measure; {undefined}"},
            {"name": f"{name}: spectral dimension at step 0",
             "measured": _round(history["spectral"][0]), "known": known,
             "note": f"start check of the measure; {undefined}"},
            {"name": f"{name}: ball dimension after {STEPS} steps",
             "measured": _round(history["ball"][-1]), "known": TARGET,
             "note": f"3 is the space-test target, not a known answer; {undefined}"},
            {"name": f"{name}: spectral dimension after {STEPS} steps",
             "measured": _round(history["spectral"][-1]), "known": TARGET,
             "note": f"3 is the space-test target, not a known answer; {undefined}"},
            {"name": f"{name}: steps across the network, start -> end",
             "measured": round(history["across"][-1], 2), "known": None,
             "note": f"start {history['across'][0]:.2f}; a 3D space of {JOINTS} joints "
                     "needs many steps, a small-world network only a handful"},
            {"name": f"{name}: connections moved since the previous checkpoint (step {history['step'][-2]})",
             "measured": history["accepted"][-1], "known": None,
             "note": "near 0 means the rewiring has stalled (no move lowers the action)"},
        ]
    return rows


def _summary(verdict: str, tracks: dict) -> str:
    lost = [name for name, history in tracks.items()
            if history["ball"][-1] is None and history["spectral"][-1] is None]
    if verdict == "pass":
        return ("At least one start settled into a network whose ball-growth and spectral "
                "dimension both read about 3, the way 3D space does.")
    text = ("No start settled into 3D space. ")
    if lost:
        text += (f"{len(lost)} of {len(tracks)} ended as a 'small world', where every joint is "
                 "only a few connections from every other, so no dimension can be read at all; "
                 "the rewiring moves keep making shortcuts instead of building a grid. ")
    control = tracks.get("3D reference (control)")
    if control and "3D reference (control)" in lost:
        peak = max(value for value in control["spectral"] if value is not None)
        text += ("Even the control that started as real 3D space was crumpled: its spectral "
                 f"dimension rose from {control['spectral'][0]:.2f} to {peak:.2f} before it "
                 "became unreadable. ")
    unread = [name for name, history in tracks.items()
              if history["ball"][0] is None and history["spectral"][0] is None]
    if unread:
        text += (f"Caveat: at {JOINTS} joints the measures cannot read {', '.join(unread)} "
                 "even at the start, so those runs show only the steps-across shrinking.")
    return text.strip() + (" Some readings were close to 3 but not both, so this is unclear."
                           if verdict == "unclear" else "")


def _meaning(tracks: dict, alone: dict) -> str:
    control = "3D reference (control)"
    mine, theirs = tracks[control], alone[control]
    at = mine["step"][-1]
    other = theirs["step"].index(at)
    crumpled = mine["ball"][-1] is None and mine["spectral"][-1] is None
    crumpled_alone = theirs["ball"][other] is None and theirs["spectral"][other] is None
    state = ("had crumpled into a 'small world' where every joint is a few connections from "
             "every other" if crumpled else
             f"still read as ball {_round(mine['ball'][-1])}, spectral {_round(mine['spectral'][-1])}")
    first = (f"With all rules on (balancing drift included) plus rewiring, the network that began as "
             f"real 3D space {state} by step {at}, its steps across falling from "
             f"{mine['across'][0]:.1f} to {mine['across'][-1]:.1f} against {theirs['across'][other]:.1f} "
             "with rewiring alone (in the picture the two lines lie on top of each other).")
    collapsed = mine["joints"][0] - mine["joints"][-1]
    why = (" (rewiring reads only which joints connect, and no joint collapsed)" if not collapsed
           else f" ({collapsed} joints collapsed)")
    second = (("So attraction, push and drift do not change the small-world outcome of the rewiring "
               "rule" if crumpled == crumpled_alone else
               "So attraction, push and drift do change the outcome of the rewiring rule")
              + why + "; the random, 4D and 6D starts ran with rewiring only, because the pull is "
              "too slow on them at this size.")
    return first + " " + second
