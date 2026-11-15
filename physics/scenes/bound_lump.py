"""Bound lump scene: the size of the biggest lump while its surroundings spread or fall.

Contract: `bound_lump(cells_per_side=8, boxes_per_cell_side=6, seed=1, unevenness=0.3,
steps=60, every=2, settings=None, device="cuda")` builds
physics.potts.new_universe(cells_per_side, boxes_per_cell_side, seed, unevenness)
(nothing placed: the lump forms by itself), runs `steps` steps with settings (default
Settings.all_on()) and at the start and every `every` steps finds the lump holding the
most energy: of the energy volumes whose energy per box is at least
settings.lump_factor * universe.rest_a (as stats counts lumps), the one with the most
energy. That may be a different cell from one sample to the next. Returns a dict (no
verdict):
  cells_per_side, boxes_per_cell_side, seed, unevenness, steps, settings (dict)
  time                 sample times
  lump_boxes           Planck boxes the lump holds (0: no lump yet)
  lump_energy          its energy (0 with no lump)
  lump_radius          RMS radius of its energy about its centre, boxes (0 with no lump)
  surroundings_radius  RMS radius of the energy outside it about that energy's centre
  lump_radius_change   last lump_radius over the first nonzero one (0 if never one)
  surroundings_change  last surroundings_radius over the first
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.potts import ENERGY_VOLUME, Settings, Universe, new_universe
from physics.scenes.box_energies import box_energies
from physics.scenes.rms_radius import rms_radius
from physics.scenes.sampled_run import sampled_run


def bound_lump(cells_per_side: int = 8, boxes_per_cell_side: int = 6, seed: int = 1, unevenness: float = 0.3,
               steps: int = 60, every: int = 2, settings: Settings | None = None, device: str = "cuda") -> dict:
    """Biggest lump's boxes, energy and radius, and the surroundings' radius, over time."""
    settings = settings or Settings.all_on()
    universe = new_universe(cells_per_side, boxes_per_cell_side, seed, unevenness, device=device)
    samples = sampled_run(universe, settings, steps, every, lambda sampled: _measure(sampled, settings))
    result = {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "seed": seed,
              "unevenness": unevenness, "steps": steps, "settings": asdict(settings),
              "time": [time for time, _ in samples]}
    for key in ("lump_boxes", "lump_energy", "lump_radius", "surroundings_radius"):
        result[key] = [measured[key] for _, measured in samples]
    radii = result["lump_radius"]
    first_radius = next((radius for radius in radii if radius > 0.0), 0.0)
    result["lump_radius_change"] = radii[-1] / first_radius if first_radius > 0.0 else 0.0
    result["surroundings_change"] = result["surroundings_radius"][-1] / result["surroundings_radius"][0]
    return result


def _measure(universe: Universe, settings: Settings) -> dict:
    a_energy, b_energy = box_energies(universe)
    energy = a_energy + b_energy
    owner = universe.owner.numpy()
    occupied = owner >= 0
    count = universe.cell_count
    held = np.bincount(owner[occupied], weights=energy[occupied], minlength=count)
    boxes = np.bincount(owner[occupied], minlength=count)
    is_lump = ((universe.kind.numpy() == ENERGY_VOLUME) & (boxes > 0)
               & (held >= settings.lump_factor * universe.rest_a * np.maximum(boxes, 1)))
    if not is_lump.any():
        return {"lump_boxes": 0.0, "lump_energy": 0.0, "lump_radius": 0.0,
                "surroundings_radius": rms_radius(energy)}
    lump = owner == int(np.argmax(np.where(is_lump, held, -1.0)))
    return {"lump_boxes": float(lump.sum()), "lump_energy": float(energy[lump].sum()),
            "lump_radius": rms_radius(energy, lump), "surroundings_radius": rms_radius(energy, ~lump)}
