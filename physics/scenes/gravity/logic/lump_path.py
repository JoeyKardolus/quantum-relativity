"""Lump path: step a universe and follow its lumps as bodies.

Contract: `lump_path(universe, settings, near, radius, steps, every, quantum=False)`
steps the universe `steps` times (`physics.engine.step`; `quantum=False` is phase 1 of
the fit, the layers below axiom 7) and, at step 0 and after every `every` steps, finds
each lump again (`lump_centre` within `radius` volumes of where it was last). Returns
  steps     the steps sampled
  centres   per lump, per sample: its centre in volumes (x, y, z), followed past the edge
  amounts   per lump, per sample: its extra energy
  collapses, tears   joints collapsed and volumes torn in the run
The run stops, with one last sample, at the step of the first
collapse or tear: a lump that fell into one cell is no body to follow, and a universe
that tears is not standing (its tears double from step to step).
Raises ValueError for steps < 0 or every < 1.
"""
from __future__ import annotations

from physics.engine import Grid, Settings, step
from physics.scenes.gravity.definitions.lump_centre import lump_centre
from physics.scenes.gravity.logic.volume_energy import volume_energy


def lump_path(universe: Grid, settings: Settings, near: list, radius: float, steps: int, every: int,
              quantum: bool = False) -> dict:
    """The lumps' centres and amounts over a run, sampled every `every` steps."""
    if steps < 0 or every < 1:
        raise ValueError(f"steps must be 0 or more and every 1 or more, got {steps} and {every}")
    places = [tuple(float(one) for one in place) for place in near]
    path = {"steps": [], "centres": [[] for _ in places], "amounts": [[] for _ in places]}
    for done in range(steps + 1):
        over = bool(universe.collapses or universe.tears)
        if done % every == 0 or over:
            energy = volume_energy(universe)
            path["steps"].append(done)
            for lump, place in enumerate(places):
                places[lump], amount = lump_centre(energy, place, radius)
                path["centres"][lump].append(places[lump])
                path["amounts"][lump].append(amount)
        if over:
            break
        if done < steps:
            step(universe, settings, quantum=quantum)
    path["collapses"] = int(universe.collapses)
    path["tears"] = int(universe.tears)
    return path
