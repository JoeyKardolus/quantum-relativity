"""Space test (milestone 5): start with no geometry, run the rules, watch the dimension.

Contract: `run(seed, out_dir)` builds four starts of JOINTS joints: a random
network (no geometry), a 6D and a 4D spatial reference, and a 3D spatial
reference as a control (is 3D space kept?). Each runs STEPS steps of the rules
with energy flow, the energy-length rule and RewIRING_ATTEMPTS rewiring attempts
per step (target degree = the start's own mean degree, since rewiring keeps the
connection count; target triangles 0, the core default guess). Drift, attraction
and repulsion are off: drift scales every length alike and cannot change a
dimension, and this test asks what the rewiring does to shape. It measures
ball-growth and spectral dimension at CHECKPOINTS, log-spaced so the fast
early change is resolved (`track`), writes
the plots into out_dir and returns the result dict. Verdict, fixed in advance:
pass if some start ends with both dimensions within 10% of 3, unclear if one of
them does, fail otherwise. Picture: the steps-across plot; meaning: two plain
sentences.
"""
from __future__ import annotations

import time

import numpy as np

from physics.experiments.space.high_dimensional import high_dimensional_reference
from physics.experiments.space.plots import plot_tracks
from physics.experiments.space.track import track
from physics.network import random_network, spatial_reference
from physics.rules import RuleSettings

JOINTS = 10_000
STEPS = 120
REWIRING_ATTEMPTS = 5_000
CHECKPOINTS = sorted({0, *np.unique(np.geomspace(1, STEPS, 16).round().astype(int)).tolist()})
ENERGY_FLOW = 0.1
TARGET = 3.0
BAND = 0.10


def run(seed: int, out_dir: str) -> dict:
    """Run the space test and return its JSON-serialisable result."""
    began = time.time()
    starts = _starts(seed)
    tracks, used = {}, {}
    for name, (network, _known) in starts.items():
        settings = RuleSettings(energy_length=1.0, energy_flow=ENERGY_FLOW,
                                rewiring=REWIRING_ATTEMPTS,
                                target_degree=float(round(network.degree().mean())))
        used[name] = {"target_degree": settings.target_degree,
                      "connections": int(network.connection_count)}
        tracks[name] = track(network, settings, STEPS, CHECKPOINTS, seed)
    plots = plot_tracks(tracks, out_dir)
    numbers = _numbers(starts, tracks)
    verdict = _verdict(tracks)
    return {
        "milestone": 5,
        "title": "Space test",
        "question": "Starting from a network with no geometry (or the wrong dimension), "
                    "do the rules with rewiring settle it into something that behaves "
                    "like 3D space?",
        "verdict": verdict,
        "summary": _summary(verdict, tracks),
        "meaning": _meaning(verdict, tracks),
        "numbers": numbers,
        "plots": plots,
        "picture": "space_steps_across.png",
        "settings": {"joints": JOINTS, "steps": STEPS, "checkpoints": CHECKPOINTS,
                     "rewiring_attempts_per_step": REWIRING_ATTEMPTS,
                     "energy_flow": ENERGY_FLOW, "energy_length": 1.0, "drift": 0.0,
                     "attraction": 0.0, "repulsion": 0.0, "target_triangles": 0.0,
                     "triangle_weight": 1.0, "per_start": used,
                     "measures": "ball_dimension and spectral_dimension, default windows, "
                                 f"seed {seed}"},
        "seed": seed,
        "seconds": round(time.time() - began, 1),
    }


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


def _meaning(verdict: str, tracks: dict) -> str:
    lost = [name for name, history in tracks.items()
            if history["ball"][-1] is None and history["spectral"][-1] is None]
    if verdict == "pass":
        return ("Rewiring alone settled at least one start into a network that reads as 3D space by both "
                "measures. So 3D space can come out of the rules here without being put in.")
    control = " (even the one that began as real 3D space)" if "3D reference (control)" in lost else ""
    first = (f"Rewiring did not settle any start into 3D space: {len(lost)} of {len(tracks)} starts{control} "
             "crumpled into a 'small world' where every joint is a few connections from every other."
             if lost else
             "Rewiring did not settle any start into 3D space, though the dimension could still be read.")
    return first + " So the rewiring rule as written builds shortcuts, not space."
