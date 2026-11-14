"""Run milestone 4 with all rules on: does drift above the balance expand sparse space while clusters hold?

Contract: `run(seed, out_dir)` builds a clustered 3D reference network and
evolves it with every rule on (`all_rules_on`: energy-length, energy flow,
drift, attraction, joint push with its limit, joint collapse; repulsion is on
but no joint carries charge; no rewiring), for each (drift excess, push limit)
pair in a small sweep. Writes the picture PNG to out_dir and returns the
JSON-serialisable result dict.

Calibration, the one allowed, made before any run on the scene's even version
(`even_version`: same connections, every one holding the mean energy):
attraction makes its pull shorten a connection by SHORTENING per step; the push
limit is `limit_multiples` x the push a joint gives at that pull (2 is the
all_rules_on house value); the balancing drift is `balancing_drift` on it. Each
run's drift is balancing drift x (1 + excess).

Pass criteria, fixed before any run (unchanged from v1 and v2): some run gives
(a) accelerating expansion of sparse space (median q < 0 over the second half),
(b) scale factor grown >= 1.5x, and (c) clusters bound at a steady size (mean
size ratio 0.8..1.25) with the network still readable at the end. A collapse is
"contained" when a cluster lost more than a tenth of its joints while sparse
space kept at least 90% of its joints.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, replace

import numpy as np

from physics.experiments.expansion.clustered_reference import clustered_reference
from physics.experiments.expansion.even_version import even_version
from physics.experiments.expansion.evolve import evolve
from physics.experiments.expansion.plots import draw_picture
from physics.experiments.expansion.result import expansion_result
from physics.measure import distances, expansion_rate
from physics.rules import all_rules_on, balancing_drift, pull

SETTINGS = {"sparse_joints": 4000, "clusters": 4, "cluster_joints": 400, "cluster_spread": 0.025,
            "neighbours": 10, "steps": 100, "measure_every": 5,
            "sparse_sources": 48, "targets_per_source": 64, "cluster_sources": 8,
            "shortening_per_background_pull": 0.01,
            "drift_excesses": [0.0, 0.1, 0.5, 2.0], "limit_multiples": [2.0, 4.0]}


def run(seed: int, out_dir: str) -> dict:
    """Calibrate, sweep, draw, judge; return the result dict."""
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    network, labels = clustered_reference(SETTINGS["sparse_joints"], SETTINGS["clusters"],
                                          SETTINGS["cluster_joints"], SETTINGS["cluster_spread"],
                                          SETTINGS["neighbours"], seed)
    rng = np.random.default_rng(seed + 1000)
    probes = _probes(network, labels, rng)
    even = even_version(network)
    background_pull = float(np.median(pull(even, even.energy, even.length)))
    base = all_rules_on(attraction=SETTINGS["shortening_per_background_pull"] / background_pull)
    balancing = balancing_drift(even, base, rng)
    start_pull = pull(network, network.energy, network.length)
    runs = []
    for excess in SETTINGS["drift_excesses"]:
        for multiple in SETTINGS["limit_multiples"]:
            settings = _settings(base, balancing, excess, multiple)
            record = evolve(network, labels, settings, SETTINGS["steps"], SETTINGS["measure_every"],
                            probes)
            runs.append(_summarise(excess, multiple, settings, record))
    picture = draw_picture(runs, out_dir)
    calibration = {"background_pull": background_pull, "balancing_drift": balancing,
                   "base_rules": asdict(base),
                   "start_pull_over_background": {
                       "sparse_median": float(np.median(start_pull[labels == -1]) / background_pull),
                       "cluster_median": float(np.median(start_pull[labels >= 0]) / background_pull)}}
    return expansion_result(runs, picture, {**SETTINGS, **calibration}, network, seed,
                            time.time() - started)


def _settings(base, balancing: float, excess: float, multiple: float):
    asked = base.joint_push * SETTINGS["shortening_per_background_pull"]
    return replace(base, drift=balancing * (1 + excess), push_limit=multiple * asked)


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


def _summarise(excess: float, multiple: float, settings, record: dict) -> dict:
    scale = np.array(record["scale"])
    spacing = np.diff(record["step"]).mean() if len(scale) > 1 else 1.0
    readable = len(scale) > 3 and np.all(np.isfinite(scale))
    measured = expansion_rate(scale, spacing) if readable else None
    sizes = np.array(record["cluster_size"])
    half = len(scale) // 2
    return {"drift_excess": excess, "limit_multiple": multiple, "drift": settings.drift,
            "push_limit": settings.push_limit, "record": record,
            "rate": measured["rate"].tolist() if readable else [],
            "median_q": float(np.nanmedian(measured["deceleration"][half:])) if readable else None,
            "early_rate": float(measured["rate"][1]) if readable else None,
            "late_rate": float(measured["rate"][-2]) if readable else None,
            "scale_ratio": float(scale[-1] / scale[0]),
            "cluster_ratio": float(np.nanmean(sizes[-1] / sizes[0])),
            "joints_lost": int(record["joints"][0] - record["joints"][-1]),
            "sparse_kept": record["sparse_by_step"][-1] / record["sparse_by_step"][0],
            "steps_run": int(record["step"][-1])}
