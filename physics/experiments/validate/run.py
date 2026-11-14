"""Validate experiment entry point: do the measurements recover known results?

Contract: `run(seed, out_dir)` runs the four parts (spatial dimension, proper
time, twin paradox, Myrheim-Meyer), writes their PNGs into out_dir and returns
the JSON-serialisable result dict (picture: the dimension plot; meaning: two
plain sentences). Verdict follows the PRD acceptance: "pass" when the 2D and
3D spatial dimensions, proper time and the twin paradox all land in the pre-set
10% band; otherwise "fail", naming the misses. 1D, 4D and Myrheim-Meyer are
extra readings: a miss there is reported as a known weak spot, not a fail.
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
        "verdict": "fail" if _required(misses) else "pass",
        "summary": _summary(misses),
        "meaning": _meaning(misses, len(numbers)),
        "numbers": numbers,
        "plots": dimension_plots + time_plots + twin_plots + order_plots,
        "picture": dimension_plots[0],
        "settings": {"tolerance": "10% relative", "dimension": dimension_settings,
                     "proper_time": time_settings, "twin_paradox": twin_settings,
                     "myrheim_meyer": order_settings, "misses": misses,
                     "acceptance": "2D and 3D dimension, proper time, twin paradox; the rest is extra"},
        "seed": seed,
        "seconds": round(time.time() - started, 1),
    }


def _required(misses: list[str]) -> list[str]:
    """The misses the PRD acceptance counts: 2D or 3D space, proper time, twin paradox."""
    return [miss for miss in misses if "2D space" in miss or "3D space" in miss
            or miss in ("proper time (longest chain)", "twin paradox")]


def _summary(misses: list[str]) -> str:
    base = ("We built networks from flat space and flat spacetime, threw the positions away, and "
            "measured dimension and elapsed time from the connections alone. ")
    if not misses:
        return base + ("Every measurement landed within 10% of the known answer, and the travelling "
                       "twin's longest causal chain came out shorter, as relativity predicts.")
    return base + (f"{len(misses)} measurement(s) missed the 10% band: {'; '.join(misses)}. "
                   "The rest matched; the misses are reported as misses, not tuned away.")


def _meaning(misses: list[str], checks: int) -> str:
    required = _required(misses)
    extra = [miss for miss in misses if miss not in required]
    twin = ("the travelling twin's clock did not reliably come out behind" if "twin paradox" in misses
            else "the travelling twin's clock came out behind, as relativity says")
    matched = ("matched flat space and spacetime" if not misses
               else f"matched flat space and spacetime in all but {len(misses)} of {checks} readings")
    first = (f"Dimension and elapsed time read off the connections alone, with every position "
             f"thrown away, {matched}, and {twin}.")
    if required:
        return first + (f" So the rulers the later tests use are not yet trustworthy: "
                        f"{'; '.join(required)} missed, reported, not tuned away.")
    named = ("the 1D readings" if all("1D space" in miss for miss in extra)
             else f"the extra readings {'; '.join(extra)}")
    weak = (f", but {named} missed the 10% band, a known weak spot, reported, not tuned away."
            if extra else ".")
    return first + (" So the network carries real geometry without any coordinates and the 2D and "
                    "3D rulers the later tests use can be trusted" + weak)
