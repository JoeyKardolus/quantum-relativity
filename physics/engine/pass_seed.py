"""Pass seed: the rounding hash's seed for one fluid pass of one substep of one step.

Contract: `pass_seed(step, sub_index, pass_index)` returns a non-negative int below
2**31: (step * 64 + pass_index) for substep 0 (the coarse step's seed, unchanged), mixed
with sub_index * 0x9E3779B1 for later substeps, so every pass of every substep rounds
with its own uniforms. Hop and leak combine it with each universe's key (state.py).
"""
from __future__ import annotations


def pass_seed(step: int, sub_index: int, pass_index: int) -> int:
    """The hash seed of pass `pass_index` in substep `sub_index` of step `step`."""
    return ((step * 64 + pass_index) ^ (sub_index * 0x9E3779B1)) & 0x7FFFFFFF
