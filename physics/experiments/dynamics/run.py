"""Grid dynamics entry point: does each rule move the network the way it says?

Contract: `run(seed, out_dir)` runs `rule_checks` and `plot_drift`, writes the
PNG into out_dir and returns the JSON-serialisable result dict. Verdict is
"pass" only when every rule check passes its band set in checks.py.
"""
from __future__ import annotations

import os
import time

from physics.experiments.dynamics.checks import rule_checks
from physics.experiments.dynamics.plots import plot_drift


def run(seed: int, out_dir: str) -> dict:
    """Run Grid dynamics and return its result dict."""
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    numbers, passed = rule_checks(seed)
    plots = [plot_drift(seed, out_dir)]
    failed = len(passed) - sum(passed)
    summary = ("Each rule was switched on alone on a small network. "
               + (f"All {len(passed)} rules (energy-length, drift, energy flow, attraction, "
                  "repulsion, rewiring) moved the network the way their definition says."
                  if not failed else f"{failed} of {len(passed)} rules did not do what they say.")
               + " This checks that the code does what the rules say, not that the rules are right.")
    return {
        "milestone": 2,
        "title": "Grid dynamics",
        "question": "Does each rule (energy-length, drift, energy flow, attraction, repulsion, "
                    "rewiring) change the network the way it is defined to?",
        "verdict": "fail" if failed else "pass",
        "summary": summary,
        "numbers": numbers,
        "plots": plots,
        "settings": {"joints": 2000, "dimension": 3, "drift": 0.01, "drift_steps": 10,
                     "energy_flow": 0.5, "attraction": 0.05, "repulsion": 0.1,
                     "rewiring_attempts": 5000},
        "seed": seed,
        "seconds": round(time.time() - started, 1),
    }
