"""Searched phase: which runs of a search are the searched points.

Contract: `searched_phase(phase)` returns True for the "random" phase and every
"refine<r>" phase: the short runs that look for a point. The defaults, the hold runs and
the sweep are not searched points.
"""
from __future__ import annotations


def searched_phase(phase: str) -> bool:
    """Whether a phase's runs are searched points."""
    return phase == "random" or phase.startswith("refine")
