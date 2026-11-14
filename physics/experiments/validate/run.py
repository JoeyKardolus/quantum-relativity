"""Validate experiment entry point: do the measurements recover known results?

Contract: `run(seed, out_dir)` runs the four parts (spatial dimension, proper
time, twin paradox, Myrheim-Meyer), writes their PNGs into out_dir and returns
the JSON-serialisable result dict. Verdict is "pass" only when every part
passes its pre-set 10% band; otherwise "fail", naming the misses.
"""
from __future__ import annotations

import time

from physics.experiments.validate.dimension import measure_dimensions
from physics.experiments.validate.proper_time import measure_proper_time
from physics.experiments.validate.spacetime_dimension import measure_spacetime_dimension
from physics.experiments.validate.twin import measure_twin_paradox


def run(seed: int, out_dir: str) -> dict:
    """Run Validate and return its result dict."""
    started = time.time()
    dimension_numbers, dimension_plots, dimension_settings = measure_dimensions(seed, out_dir)
    time_numbers, time_plots, time_settings, time_passed = measure_proper_time(seed, out_dir)
    twin_numbers, twin_plots, twin_settings, twin_passed = measure_twin_paradox(seed, out_dir)
    order_numbers, order_plots, order_settings = measure_spacetime_dimension(seed, out_dir)
    numbers = dimension_numbers + time_numbers + twin_numbers + order_numbers
    misses = [entry["name"] for entry in dimension_numbers + order_numbers
              if entry["note"].count("MISS")]
    if not time_passed:
        misses.append("proper time (longest chain)")
    if not twin_passed:
        misses.append("twin paradox")
    return {
        "milestone": 1,
        "title": "Validate",
        "question": "Can distance, dimension and proper time be read off the network alone, "
                    "and do they match the known answers of flat space and spacetime?",
        "verdict": "fail" if misses else "pass",
        "summary": _summary(misses),
        "numbers": numbers,
        "plots": dimension_plots + time_plots + twin_plots + order_plots,
        "settings": {"tolerance": "10% relative", "dimension": dimension_settings,
                     "proper_time": time_settings, "twin_paradox": twin_settings,
                     "myrheim_meyer": order_settings, "misses": misses},
        "seed": seed,
        "seconds": round(time.time() - started, 1),
    }


def _summary(misses: list[str]) -> str:
    base = ("We built networks from flat space and flat spacetime, threw the positions away, and "
            "measured dimension and elapsed time from the connections alone. ")
    if not misses:
        return base + ("Every measurement landed within 10% of the known answer, and the travelling "
                       "twin's longest causal chain came out shorter, as relativity predicts.")
    return base + (f"{len(misses)} measurement(s) missed the 10% band: {'; '.join(misses)}. "
                   "The rest matched; the misses are reported as misses, not tuned away.")
