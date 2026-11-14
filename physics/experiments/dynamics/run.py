"""Grid dynamics entry point: does each rule move the network the way it says?

Contract: `run(seed, out_dir)` runs `rule_checks`, `plot_drift` and
`plot_pull_fade`, writes the PNGs into out_dir and returns the
JSON-serialisable result dict (picture: the pull-fade plot). Verdict is "pass"
only when every rule check passes its band set in checks.py.
"""
from __future__ import annotations

import os
import time

from physics.experiments.dynamics.checks import rule_checks
from physics.experiments.dynamics.fade_plot import plot_pull_fade
from physics.experiments.dynamics.plots import plot_drift
from physics.experiments.dynamics.pull_fade import pull_by_hops


def run(seed: int, out_dir: str) -> dict:
    """Run Grid dynamics and return its result dict."""
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    profile = pull_by_hops(seed)
    numbers, passed = rule_checks(seed, profile)
    picture = plot_pull_fade(profile, out_dir)
    plots = [picture, plot_drift(seed, out_dir)]
    failed = len(passed) - sum(passed)
    summary = ("Each rule was switched on alone on a small network. "
               + (f"All {len(passed)} checks (energy-length, drift, energy flow, attraction, pull "
                  "fade, joint push cap, joint collapse energy, repulsion, rewiring) came out the "
                  "way the rule's definition says."
                  if not failed else f"{failed} of {len(passed)} checks did not do what the rule says.")
               + " This checks that the code does what the rules say, not that the rules are right.")
    meaning = _meaning(failed, len(passed), profile)
    return {
        "milestone": 2,
        "title": "Grid dynamics",
        "question": "Does each rule (energy-length, drift, energy flow, attraction with its pull, "
                    "joint push and joint collapse, repulsion, rewiring) change the network the "
                    "way it is defined to?",
        "verdict": "fail" if failed else "pass",
        "summary": summary,
        "numbers": numbers,
        "meaning": meaning,
        "plots": plots,
        "picture": picture,
        "settings": {"joints": 2000, "dimension": 3, "drift": 0.01, "drift_steps": 10,
                     "energy_flow": 0.5, "attraction": 0.05, "repulsion": 0.1,
                     "rewiring_attempts": 5000, "joint_push": 0.8, "push_limit": 1.0,
                     "collapse_share": 0.05, "pull_fade_profile": profile},
        "seed": seed,
        "seconds": round(time.time() - started, 1),
    }


def _meaning(failed: int, checks: int, profile: dict) -> str:
    excess = profile["excess"]
    first = (f"All {checks} rule checks passed: the v2 pull is strongest at a packed lump and falls "
             f"with every connection in between (to {excess[-1] / excess[0]:.0%} of its centre value "
             f"{profile['hops'][-1]} connections out), a joint never pushes past its limit, and joint "
             "collapse keeps every bit of energy."
             if not failed else f"{failed} of {checks} rule checks failed, so some rule does not do what it says.")
    share = excess[0] / profile["background"]
    if share >= 1:
        return first + f" A packed lump more than doubles the pull at its centre ({share:.0%} extra)."
    return (first + f" But a packed lump adds only {share:.0%} to the pull every joint already feels, "
            "so the pull mostly squeezes all of space at once; that is a property of the rule as "
            "written, which the gravity and expansion tests then run into.")
