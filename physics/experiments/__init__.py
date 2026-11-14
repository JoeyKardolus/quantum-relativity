"""Experiments layer: one subpackage per milestone, each with run(seed, out_dir) -> dict.

Module map:
  validate/   milestone 1: dimension, proper time and twin paradox read off reference networks
  dynamics/   milestone 2: each rule moves the network the way it is defined to
  gravity/    milestone 3: with attraction off, do lumps of energy approach each other?
  expansion/  milestone 4: does drift expand space while dense clusters stay bound?
  space/      milestone 5: does rewiring settle a network into 3D space?

A result dict has milestone, title, question, verdict (pass/fail/unclear),
summary, numbers [{name, measured, known, note}], plots (file names in out_dir),
settings, seed, seconds. run_all(seed, out_dir) runs milestones 1-5 in order, each
writing PNGs into out_dir/<name>, and returns their dicts with "plot_dir" set to that folder.
"""
from __future__ import annotations

import os

from physics.experiments import dynamics, expansion, gravity, space, validate

_EXPERIMENTS = (("validate", validate), ("dynamics", dynamics), ("gravity", gravity),
                ("expansion", expansion), ("space", space))


def run_all(seed: int, out_dir: str) -> list[dict]:
    """Run milestones 1-5; return their result dicts in order."""
    results = []
    for name, experiment in _EXPERIMENTS:
        plot_dir = os.path.join(out_dir, name)
        os.makedirs(plot_dir, exist_ok=True)
        result = experiment.run(seed, plot_dir)
        result["plot_dir"] = plot_dir
        results.append(result)
    return results


__all__ = ["run_all"]
