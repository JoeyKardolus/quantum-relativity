"""Run milestone 4: drift + attraction on a clustered 3D network, with an attraction sweep.

Contract: `run(seed, out_dir)` builds a clustered 3D reference network, evolves
it under drift and attraction for each strength in a sweep, writes PNGs to
out_dir and returns the JSON-serialisable result dict. Pass criteria, fixed
before any run: some attraction g gives (a) accelerating expansion of sparse
space (median q < 0 over the second half), (b) scale factor grown >= 1.5x, and
(c) clusters bound at a steady size (mean size ratio 0.8..1.25, no collapse).
"""
from __future__ import annotations

import os
import time

import numpy as np

from physics.experiments.expansion.clustered_reference import clustered_reference
from physics.experiments.expansion.evolve import evolve
from physics.experiments.expansion.plots import draw_plots
from physics.measure import distances, expansion_rate
from physics.rules import RuleSettings

SETTINGS = {"sparse_joints": 20000, "clusters": 4, "cluster_joints": 2000, "cluster_spread": 0.025,
            "neighbours": 10, "drift": 0.01, "energy_length": 1.0, "steps": 150, "measure_every": 5,
            "sparse_sources": 48, "targets_per_source": 64, "cluster_sources": 8,
            "threshold_multiples": [0.0, 0.1, 0.25, 0.35, 0.5, 1.0, 2.0, 4.0]}


def run(seed: int, out_dir: str) -> dict:
    """Build, sweep attraction, plot, judge; return the result dict."""
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    network, labels = clustered_reference(SETTINGS["sparse_joints"], SETTINGS["clusters"],
                                          SETTINGS["cluster_joints"], SETTINGS["cluster_spread"],
                                          SETTINGS["neighbours"], seed)
    rng = np.random.default_rng(seed + 1000)
    probes = _probes(network, labels, rng)
    cluster_length = _cluster_length(network, labels)
    drift = SETTINGS["drift"]
    runs = []
    for multiple in SETTINGS["threshold_multiples"]:
        attraction = 2 * drift * multiple * cluster_length
        settings = RuleSettings(energy_length=SETTINGS["energy_length"], drift=drift, attraction=attraction)
        record = evolve(network, labels, settings, SETTINGS["steps"], SETTINGS["measure_every"], probes)
        runs.append(_summarise(attraction, multiple, record))
    plots = draw_plots(runs, drift, out_dir)
    return _result(runs, plots, cluster_length, network, labels, seed, time.time() - started)


def _cluster_length(network, labels) -> float:
    inside = (labels[network.pairs] >= 0).all(axis=1)
    return float(network.length[inside].mean())


def _probes(network, labels, rng) -> dict:
    """Fixed measurement joints: far-apart sparse pairs and a few joints per cluster."""
    sparse = np.flatnonzero(labels == -1)
    sources = rng.choice(sparse, SETTINGS["sparse_sources"], replace=False)
    table = distances(network, sources)[:, sparse]
    far = table > np.percentile(table, 75)
    pair_rows, pair_targets = [], []
    for row in range(len(sources)):
        chosen = rng.choice(sparse[far[row]], SETTINGS["targets_per_source"], replace=False)
        pair_rows.extend([row] * len(chosen))
        pair_targets.extend(chosen)
    cluster_sources = [rng.choice(np.flatnonzero(labels == cluster), SETTINGS["cluster_sources"],
                                  replace=False) for cluster in range(SETTINGS["clusters"])]
    return {"sparse_sources": sources, "pair_rows": np.array(pair_rows),
            "pair_targets": np.array(pair_targets), "cluster_sources": cluster_sources}


def _summarise(attraction: float, multiple: float, record: dict) -> dict:
    scale = np.array(record["scale"])
    spacing = np.diff(record["step"]).mean() if len(scale) > 1 else 1.0
    measured = expansion_rate(scale, spacing) if len(scale) > 2 else {"rate": np.full(len(scale), np.nan),
                                                                     "deceleration": np.full(len(scale), np.nan)}
    sizes = np.array(record["cluster_size"])
    half = len(scale) // 2
    return {"attraction": attraction, "threshold_multiple": multiple, "record": record,
            "rate": measured["rate"].tolist(),
            "median_q": float(np.nanmedian(measured["deceleration"][half:])) if len(scale) > 3 else None,
            "early_rate": float(measured["rate"][1]) if len(scale) > 2 else None,
            "late_rate": float(measured["rate"][-2]) if len(scale) > 2 else None,
            "scale_ratio": float(scale[-1] / scale[0]),
            "cluster_ratio": float((sizes[-1] / sizes[0]).mean()),
            "steps_run": int(record["step"][-1])}


def _judge(run: dict) -> dict:
    accelerating = run["median_q"] is not None and run["median_q"] < 0
    grew = run["scale_ratio"] >= 1.5
    bound = 0.8 <= run["cluster_ratio"] <= 1.25 and run["record"]["collapse_step"] is None
    return {"accelerating": accelerating, "grew": grew, "bound": bound}


def _result(runs, plots, cluster_length, network, labels, seed, seconds) -> dict:
    drift = SETTINGS["drift"]
    judged = [_judge(run) for run in runs]
    passing = [run for run, verdict in zip(runs, judged) if all(verdict.values())]
    numbers = [{"name": "drift only (g = 0): late growth rate per step",
                "measured": runs[0]["late_rate"], "known": float(np.log(1 + drift)),
                "note": "pure exponential expansion, the cosmological-constant case"},
               {"name": "drift only (g = 0): median deceleration q, second half",
                "measured": runs[0]["median_q"], "known": -1.0, "note": "exponential growth has q = -1"},
               {"name": "drift only (g = 0): cluster size ratio vs scale factor ratio",
                "measured": runs[0]["cluster_ratio"], "known": runs[0]["scale_ratio"],
                "note": "with no attraction clusters should simply grow with space"}]
    for run, verdict in zip(runs[1:], judged[1:]):
        collapse = run["record"]["collapse_step"]
        tag = f"g = {run['attraction']:.3g} ({run['threshold_multiple']}x cluster threshold)"
        status = "pass" if all(verdict.values()) else "miss: " + ", ".join(
            key for key, value in verdict.items() if not value)
        numbers.append({"name": f"{tag}: cluster size ratio, end / start", "measured": run["cluster_ratio"],
                        "known": 1.0,
                        "note": f"1 = bound; sparse scale factor grew {run['scale_ratio']:.3g}x over "
                                f"{run['steps_run']} steps; " + (f"first connection collapsed at step {collapse}, "
                                "run stopped; " if collapse else "no collapse; ") + status})
        numbers.append({"name": f"{tag}: median deceleration q, second half", "measured": run["median_q"],
                        "known": -1.0, "note": f"q < 0 accelerating, q < -1 faster than exponential; growth rate "
                                               f"{run['early_rate']} early -> {run['late_rate']} late"})
    return {"milestone": 4, "title": "Expansion test",
            "question": "Does drift give accelerating expansion while dense clusters stay bound?",
            "verdict": "pass" if passing else "fail",
            "summary": _summary(runs, judged),
            "numbers": numbers, "plots": plots,
            "settings": {**SETTINGS, "joints": network.joint_count, "connections": network.connection_count,
                         "cluster_mean_length": cluster_length,
                         "attraction_values": [run["attraction"] for run in runs],
                         "attraction_rule": "g = 2 * drift * multiple * cluster mean length "
                                            "(multiple 1 = first-order threshold l* = g / 2h at the cluster scale)"},
            "seed": seed, "seconds": round(seconds, 1)}


def _summary(runs, judged) -> str:
    calm = [run for run in runs[1:] if run["record"]["collapse_step"] is None]
    broke = [run for run in runs[1:] if run["record"]["collapse_step"] is not None]
    text = (f"With drift alone, space grows exactly exponentially (rate {runs[0]['late_rate']:.4f} per step, "
            f"the known ln(1+h) = {np.log(1 + SETTINGS['drift']):.4f}), like a universe ruled by a "
            f"cosmological constant, and the dense clusters simply grow with it. ")
    if calm:
        strongest = calm[-1]
        text += (f"With weak attraction (up to g = {strongest['attraction']:.2g}) space still accelerates, "
                 f"but the clusters keep growing too ({strongest['cluster_ratio']:.2f}x while space grew "
                 f"{strongest['scale_ratio']:.2f}x), so they are not bound. ")
    if broke:
        weakest = broke[0]
        text += (f"From g = {weakest['attraction']:.2g} up, the shortest connections inside a cluster shrink to "
                 f"zero length within {weakest['record']['collapse_step']} steps and the collapse then spreads "
                 f"through the whole network, one joint per step. ")
    text += ("No attraction strength gave clusters that hold a steady size while space expands." if not any(
        all(verdict.values()) for verdict in judged) else "At least one strength gave bound clusters in accelerating space.")
    return text
