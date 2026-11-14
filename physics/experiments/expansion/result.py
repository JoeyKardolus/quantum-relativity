"""The expansion test's result dict: verdict, numbers, summary and the two meaning sentences.

Contract: `expansion_result(runs, picture, settings, network, seed, seconds)`
judges every run against the pass criteria in run.py and returns the
JSON-serialisable result dict. A run's collapse is "contained" when some
cluster lost more than COLLAPSED_SHARE of its joints while sparse space kept at
least 1 - COLLAPSED_SHARE of its joints to the end.
"""
from __future__ import annotations

import numpy as np

COLLAPSED_SHARE = 0.1
V2 = {"cluster_ratio": 1.54, "scale_ratio": 2.13, "runaway_steps": 4, "joints_lost": 5416, "joints": 5600}


def expansion_result(runs: list[dict], picture: str, settings: dict, network, seed: int,
                     seconds: float) -> dict:
    """Judge the runs and build the result dict."""
    for entry in runs:
        entry["judged"] = _judge(entry)
        entry["clusters_collapsed"] = _clusters_collapsed(entry)
        entry["collapse_contained"] = entry["clusters_collapsed"] > 0 and entry["sparse_kept"] >= 1 - COLLAPSED_SHARE
    passing = [entry for entry in runs if all(entry["judged"].values())]
    return {"milestone": 4, "title": "Expansion test",
            "question": "With all rules on and drift a little above the balance, does sparse space "
                        "expand faster and faster while dense clusters stay bound, and does a cluster "
                        "past the push limit collapse without taking all of space with it?",
            "verdict": "pass" if passing else "fail",
            "summary": _summary(runs), "meaning": _meaning(runs), "numbers": _numbers(runs, settings),
            "plots": [picture], "picture": picture,
            "settings": {**settings, "joints": network.joint_count, "connections": network.connection_count,
                         "runs": [_tag(entry) for entry in runs]},
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
    return f"drift = balance {entry['drift_excess']:+.0%}, push limit {entry['limit_multiple']:g}x"


def _ended(entry: dict) -> str:
    breakdown = entry["record"]["breakdown_step"]
    return (f"broke down at step {breakdown} ({entry['record']['breakdown_reason']})"
            if breakdown else f"ran all {entry['steps_run']} steps")


def _numbers(runs: list[dict], settings: dict) -> list[dict]:
    numbers = [{"name": "calibration: balancing drift on the scene's even version",
                "measured": settings["balancing_drift"], "known": None,
                "note": "drift that keeps the even version the same size for one step"},
               {"name": "start pull of sparse joints / background pull (median)",
                "measured": settings["start_pull_over_background"]["sparse_median"], "known": 1.0,
                "note": "below 1 sparse space should start expanding under the balancing drift"},
               {"name": "start pull of cluster joints / background pull (median)",
                "measured": settings["start_pull_over_background"]["cluster_median"], "known": None,
                "note": "above the push limit multiple / 1 the cluster joints start past the limit"}]
    for entry in runs:
        status = "pass" if all(entry["judged"].values()) else "miss: " + ", ".join(
            key for key, value in entry["judged"].items() if not value)
        numbers.append({"name": f"{_tag(entry)}: sparse-space scale factor, end / start",
                        "measured": entry["scale_ratio"], "known": None,
                        "note": f"growth rate {entry['early_rate']} early, {entry['late_rate']} late; "
                                f"median q {entry['median_q']} (q < 0 accelerating); {_ended(entry)}"})
        numbers.append({"name": f"{_tag(entry)}: cluster size, end / start",
                        "measured": entry["cluster_ratio"], "known": 1.0, "note": f"1 = bound; {status}"})
        numbers.append({"name": f"{_tag(entry)}: share of sparse joints still standing",
                        "measured": entry["sparse_kept"], "known": 1.0,
                        "note": f"{entry['clusters_collapsed']} cluster(s) lost over {COLLAPSED_SHARE:.0%} "
                                f"of their joints; {entry['joints_lost']} joints lost in all"})
    return numbers


def _summary(runs: list[dict]) -> str:
    return " ".join(f"{_tag(entry)}: space x{entry['scale_ratio']:.2f}, clusters x{entry['cluster_ratio']:.2f}, "
                    f"sparse joints kept {entry['sparse_kept']:.0%}, {_ended(entry)}." for entry in runs)


def _meaning(runs: list[dict]) -> str:
    return f"{_expansion_sentence(runs)} {_collapse_sentence(runs)}"


def _expansion_sentence(runs: list[dict]) -> str:
    """The smallest drift excess at which sparse space grew, against the smaller ones, and its clusters."""
    growing = sorted((entry for entry in runs
                      if entry["record"]["breakdown_step"] is None and entry["scale_ratio"] > 1),
                     key=lambda entry: entry["drift_excess"])
    if not growing:
        return ("With all rules on, sparse space shrank and fell in at every drift excess tried, because "
                "the pull of the shrinking clusters reaches nearly the whole network, so the Einstein-like "
                "balance tips toward collapse.")
    first = growing[0]
    failed = [entry["drift_excess"] for entry in runs if entry["limit_multiple"] == first["limit_multiple"]
              and entry["drift_excess"] < first["drift_excess"]]
    below = f"shrank and fell in at drift up to {max(failed):+.0%} above the balance, and " if failed else ""
    pace = (f"faster and faster (median q {first['median_q']:.1f})"
            if first["median_q"] is not None and first["median_q"] < 0 else "but not faster and faster")
    window = "inside" if 0.8 <= first["cluster_ratio"] <= 1.25 else "outside"
    return (f"With all rules on, sparse space {below}grew only at {first['drift_excess']:+.0%}: "
            f"{first['scale_ratio']:.2f}x in {first['steps_run']} steps, {pace}, while its clusters grew "
            f"{first['cluster_ratio']:.2f}x, {window} the 0.8 to 1.25 window counted as bound (v2: clusters "
            f"{V2['cluster_ratio']}x while space grew {V2['scale_ratio']}x).")


def _collapse_sentence(runs: list[dict]) -> str:
    """At the house push limit, where clusters start past it: did collapse stay inside them (v2: no)."""
    house = [entry for entry in runs if entry["limit_multiple"] == min(item["limit_multiple"] for item in runs)]
    contained = [entry for entry in house if entry["collapse_contained"]]
    if contained:
        return (f"Clusters started past the house push limit collapsed while sparse space kept at least "
                f"{1 - COLLAPSED_SHARE:.0%} of its joints in {len(contained)} of {len(house)} runs, unlike "
                f"v2 where the collapse took {V2['joints_lost']} of {V2['joints']} joints within "
                f"{V2['runaway_steps']} steps.")
    ends = [entry["record"]["breakdown_step"] or entry["steps_run"] for entry in house]
    kept = max(entry["sparse_kept"] for entry in house)
    return (f"Clusters started past the house push limit still took all of space with them within "
            f"{min(ends)} to {max(ends)} steps at every drift (at most {kept:.0%} of sparse joints left "
            f"standing; v2: {V2['runaway_steps']} steps), because a collapsing cluster raises the pull "
            "across nearly the whole network and pushes every joint past its limit.")
