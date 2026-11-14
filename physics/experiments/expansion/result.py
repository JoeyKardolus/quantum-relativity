"""The expansion test's result dict: verdict, numbers, summary and meaning.

Contract: `expansion_result(runs, plots, picture, settings, network, cluster_pull,
seed, seconds)` judges every run against the pass criteria in run.py and
returns the JSON-serialisable result dict. A run's clusters "collapsed" when
any cluster lost more than COLLAPSED_SHARE of its joints to joint collapse.
"""
from __future__ import annotations

import numpy as np

COLLAPSED_SHARE = 0.1
V1_BEST = {"cluster_ratio": 3.199, "scale_ratio": 4.21}   # v1 page, seed 1: tightest run that did not break down


def expansion_result(runs: list[dict], plots: list[str], picture: str, settings: dict, network,
                     cluster_pull: float, seed: int, seconds: float) -> dict:
    """Judge the runs and build the result dict."""
    for entry in runs:
        entry["judged"] = _judge(entry)
        entry["clusters_collapsed"] = _clusters_collapsed(entry)
    passing = [entry for entry in runs if all(entry["judged"].values())]
    return {"milestone": 4, "title": "Expansion test",
            "question": "Does drift give accelerating expansion while dense clusters, held by the "
                        "pull and joint push, stay bound?",
            "verdict": "pass" if passing else "fail",
            "summary": _summary(runs), "meaning": _meaning(runs), "numbers": _numbers(runs, settings),
            "plots": plots, "picture": picture,
            "settings": {**settings, "joints": network.joint_count, "connections": network.connection_count,
                         "median_cluster_pull": cluster_pull,
                         "runs": [_tag(entry) for entry in runs],
                         "strength_rule": "g = attraction_multiple * drift / median cluster pull; "
                                          "push_limit = limit_multiple * joint_push * attraction_multiple * drift"},
            "seed": seed, "seconds": round(seconds, 1)}


def _judge(entry: dict) -> dict:
    accelerating = entry["median_q"] is not None and entry["median_q"] < 0
    grew = entry["scale_ratio"] >= 1.5
    bound = 0.8 <= entry["cluster_ratio"] <= 1.25 and entry["record"]["breakdown_step"] is None
    return {"accelerating": accelerating, "grew": grew, "bound": bound}


def _clusters_collapsed(entry: dict) -> int:
    counts = np.array(entry["record"]["cluster_joints"])
    return int(np.sum(counts[-1] < (1 - COLLAPSED_SHARE) * counts[0]))


def _tag(entry: dict) -> str:
    if not entry["attraction_multiple"]:
        return "drift only"
    return (f"pull x{entry['attraction_multiple']:g}, limit x{entry['limit_multiple']:g}, "
            f"joint push {entry['joint_push']:g}")


def _numbers(runs: list[dict], settings: dict) -> list[dict]:
    drift = settings["drift"]
    first = runs[0]
    numbers = [{"name": "drift only: late growth rate per step", "measured": first["late_rate"],
                "known": float(np.log(1 + drift)), "note": "pure exponential expansion"},
               {"name": "drift only: median deceleration q, second half", "measured": first["median_q"],
                "known": -1.0, "note": "exponential growth has q = -1"},
               {"name": "drift only: cluster size ratio vs scale factor ratio",
                "measured": first["cluster_ratio"], "known": first["scale_ratio"],
                "note": "with no pull clusters should simply grow with space"}]
    for entry in runs[1:]:
        breakdown = entry["record"]["breakdown_step"]
        status = "pass" if all(entry["judged"].values()) else "miss: " + ", ".join(
            key for key, value in entry["judged"].items() if not value)
        ended = (f"broke down at step {breakdown} ({entry['record']['breakdown_reason']})"
                 if breakdown else f"ran all {entry['steps_run']} steps")
        numbers.append({"name": f"{_tag(entry)}: cluster size ratio, end / start",
                        "measured": entry["cluster_ratio"], "known": 1.0,
                        "note": f"1 = bound; sparse scale factor grew {entry['scale_ratio']:.3g}x; {ended}; "
                                f"{status}"})
        numbers.append({"name": f"{_tag(entry)}: joints lost to joint collapse",
                        "measured": entry["joints_lost"], "known": None,
                        "note": f"{entry['clusters_collapsed']} of {settings['clusters']} clusters lost over "
                                f"{COLLAPSED_SHARE:.0%} of their joints"})
        numbers.append({"name": f"{_tag(entry)}: median deceleration q, second half",
                        "measured": entry["median_q"], "known": -1.0,
                        "note": f"q < 0 accelerating; growth rate {entry['early_rate']} early -> "
                                f"{entry['late_rate']} late"})
    return numbers


def _followed(entry: dict) -> float:
    """Share of the expansion a cluster followed: ln(cluster growth) / ln(space growth)."""
    return float(np.log(entry["cluster_ratio"]) / np.log(entry["scale_ratio"]))


def _summary(runs: list[dict]) -> str:
    return " ".join(f"{_tag(entry)}: space x{entry['scale_ratio']:.2f}, clusters x{entry['cluster_ratio']:.2f}, "
                    f"{entry['joints_lost']} joints collapsed"
                    + (f", broke down at step {entry['record']['breakdown_step']}."
                       if entry["record"]["breakdown_step"] else ".") for entry in runs)


def _meaning(runs: list[dict]) -> str:
    calm = [entry for entry in runs[1:] if entry["record"]["breakdown_step"] is None]
    broke = [entry for entry in runs[1:] if entry["record"]["breakdown_step"] is not None]
    passing = [entry for entry in runs if all(entry["judged"].values())]
    if passing:
        best = passing[0]
        first = (f"With {_tag(best)}, sparse space grew {best['scale_ratio']:.2f}x while the clusters "
                 f"held at {best['cluster_ratio']:.2f}x their size, the first run to do so (v1: none).")
    elif calm:
        tightest = min(calm, key=_followed)
        first = (f"No run gave bound clusters: at best ({_tag(tightest)}) the clusters still grew "
                 f"{tightest['cluster_ratio']:.2f}x while space grew {tightest['scale_ratio']:.2f}x, so "
                 f"they followed {_followed(tightest):.0%} of the expansion (v1 at best {_followed(V1_BEST):.0%}): "
                 "the pull slows them but they still ride the expansion.")
    else:
        first = "Every run with the pull on broke down, so no cluster held a steady size (v1: none did either)."
    past = [entry for entry in broke if entry["limit_multiple"] is not None and entry["limit_multiple"] < 1]
    if past:
        latest = max(entry["record"]["breakdown_step"] for entry in past)
        second = (f"Clusters started past the push limit did collapse, but the collapse ran away through "
                  f"the whole network within {latest} steps, sparse space included, so a collapsing "
                  "cluster cannot yet be told apart from all of space falling in.")
    else:
        collapsed = sum(entry["clusters_collapsed"] for entry in runs)
        second = (f"Clusters past the push limit lost joints in {collapsed} cluster(s) without the "
                  "network breaking down.")
    return f"{first} {second}"
