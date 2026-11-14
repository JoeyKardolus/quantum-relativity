"""Run milestone 4: drift, pull and joint push on a clustered 3D network, with a small sweep.

Contract: `run(seed, out_dir)` builds a clustered 3D reference network and
evolves it under drift alone, then under drift + pull + joint push for each
(attraction, push limit) pair in a small sweep, plus one run with full joint
push. Writes PNGs to out_dir (picture: the network around one cluster before
and after the picture run) and returns the JSON-serialisable result dict.

Strengths are set from the network before any run: attraction g makes the pull
at the median cluster joint shorten its connections by `attraction_multiples`
x drift per step; the push limit is `limit_multiples` x the push the median
cluster joint asks for (below 1: clusters start past the limit; above 1: under it).

Pass criteria, fixed before any run: some run gives (a) accelerating expansion
of sparse space (median q < 0 over the second half), (b) scale factor grown
>= 1.5x, and (c) clusters bound at a steady size (mean size ratio 0.8..1.25)
with the network still readable at the end. A cluster "collapsed" when the run
lost more than a tenth of a cluster's joints to joint collapse.
"""
from __future__ import annotations

import os
import time

import numpy as np

from physics.experiments.expansion.clustered_reference import clustered_reference
from physics.experiments.expansion.evolve import evolve
from physics.experiments.expansion.plots import draw_plots
from physics.experiments.expansion.result import expansion_result
from physics.experiments.picture import network_picture
from physics.measure import distances, expansion_rate
from physics.rules import RuleSettings, pull

SETTINGS = {"sparse_joints": 4000, "clusters": 4, "cluster_joints": 400, "cluster_spread": 0.025,
            "neighbours": 10, "drift": 0.01, "energy_length": 1.0, "steps": 100, "measure_every": 5,
            "sparse_sources": 48, "targets_per_source": 64, "cluster_sources": 8,
            "joint_push": 0.5, "attraction_multiples": [1.0, 3.0], "limit_multiples": [0.5, 2.0],
            "full_push_run": {"joint_push": 1.0, "attraction_multiple": 3.0, "limit_multiple": 2.0},
            "picture_run": {"joint_push": 0.5, "attraction_multiple": 3.0, "limit_multiple": 2.0},
            "picture_region_cluster_sizes": 5.0}


def run(seed: int, out_dir: str) -> dict:
    """Build, sweep, plot, judge; return the result dict."""
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    network, labels = clustered_reference(SETTINGS["sparse_joints"], SETTINGS["clusters"],
                                          SETTINGS["cluster_joints"], SETTINGS["cluster_spread"],
                                          SETTINGS["neighbours"], seed)
    rng = np.random.default_rng(seed + 1000)
    probes = _probes(network, labels, rng)
    cluster_pull = float(np.median(pull(network, network.energy, network.length)[labels >= 0]))
    runs = []
    for choice in _choices():
        settings = _settings(choice, cluster_pull)
        record, final, carried = evolve(network, labels, settings, SETTINGS["steps"],
                                        SETTINGS["measure_every"], probes)
        runs.append({**_summarise(choice, settings, record), "final": final, "carried": carried})
    plots = draw_plots(runs, SETTINGS["drift"], out_dir)
    picture = _picture(out_dir, network, labels, probes, runs)
    for entry in runs:
        del entry["final"], entry["carried"]
    return expansion_result(runs, plots, picture, SETTINGS, network, cluster_pull, seed,
                            time.time() - started)


def _choices() -> list[dict]:
    """Drift alone, the sweep at the house joint push, then the full-push run."""
    sweep = [{"joint_push": SETTINGS["joint_push"], "attraction_multiple": attraction,
              "limit_multiple": limit}
             for attraction in SETTINGS["attraction_multiples"] for limit in SETTINGS["limit_multiples"]]
    return [{"joint_push": 0.0, "attraction_multiple": 0.0, "limit_multiple": None},
            *sweep, SETTINGS["full_push_run"]]


def _settings(choice: dict, cluster_pull: float) -> RuleSettings:
    drift = SETTINGS["drift"]
    if not choice["attraction_multiple"]:
        return RuleSettings(energy_length=SETTINGS["energy_length"], drift=drift)
    attraction = choice["attraction_multiple"] * drift / cluster_pull
    asked = choice["joint_push"] * choice["attraction_multiple"] * drift
    return RuleSettings(energy_length=SETTINGS["energy_length"], drift=drift, attraction=attraction,
                        joint_push=choice["joint_push"], push_limit=choice["limit_multiple"] * asked)


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


def _summarise(choice: dict, settings: RuleSettings, record: dict) -> dict:
    scale = np.array(record["scale"])
    spacing = np.diff(record["step"]).mean() if len(scale) > 1 else 1.0
    measured = expansion_rate(scale, spacing) if len(scale) > 2 else {"rate": np.full(len(scale), np.nan),
                                                                     "deceleration": np.full(len(scale), np.nan)}
    sizes = np.array(record["cluster_size"])
    half = len(scale) // 2
    return {**choice, "attraction": settings.attraction, "push_limit": settings.push_limit,
            "record": record, "rate": measured["rate"].tolist(),
            "median_q": float(np.nanmedian(measured["deceleration"][half:])) if len(scale) > 3 else None,
            "early_rate": float(measured["rate"][1]) if len(scale) > 2 else None,
            "late_rate": float(measured["rate"][-2]) if len(scale) > 2 else None,
            "scale_ratio": float(scale[-1] / scale[0]),
            "cluster_ratio": float((sizes[-1] / sizes[0]).mean()),
            "joints_lost": int(record["joints"][0] - record["joints"][-1]),
            "steps_run": int(record["step"][-1])}


def _picture(out_dir: str, network, labels, probes: dict, runs: list[dict]) -> str:
    """Cluster 0 and the sparse space around it, at the start and at the picture run's end."""
    wanted = SETTINGS["picture_run"]
    chosen = next(entry for entry in runs if all(entry[key] == value for key, value in wanted.items()))
    centre = probes["cluster_sources"][0][0]
    from_centre = distances(network, [centre])[0]
    size = from_centre[labels == 0].mean()
    region = np.flatnonzero(from_centre <= SETTINGS["picture_region_cluster_sizes"] * size)
    final, carried = chosen["final"], chosen["carried"]
    name = "network_before_after.png"
    network_picture(os.path.join(out_dir, name),
                    f"One dense cluster in expanding space: drift + pull (g x{wanted['attraction_multiple']:g}) "
                    f"+ joint push {wanted['joint_push']:g} (2D layout for drawing only)",
                    [(f"step 0: {len(region)} joints shown", network, region),
                     (f"step {chosen['steps_run']}", final, np.unique(carried[region]))],
                    [carried[region]])
    return name
