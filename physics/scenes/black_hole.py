"""Black hole scene: the sealed region's radius against the energy that collapsed.

Contract: `black_hole(side=48, energies=(250.0, 500.0, 1000.0, 2000.0), radius=6.0,
steps=40, settings=None, device="cuda")` runs one start per energy: an even ball of
that much A-energy (and its joint share of B, blob_start.py) at rest in the middle of
empty space, falling in on itself by its own pull for `steps` steps with settings
(default Settings.all_on()). After the run it measures the sealed region round the
most squeezed box (sealed_region.py). Returns a dict (no verdict):
  side, radius, steps, settings (dict), energies
  sealed_boxes, sealed_radius, sealed_energy, collapsed_energy   one per energy
  max_compression        stats "max_compression" at the end, one per energy
  radius_per_energy      sealed_radius / collapsed_energy per run (0 with nothing collapsed);
                         relativity's horizon keeps this constant
  radius_energy_power    least-squares slope of ln sealed_radius against ln collapsed_energy
                         over the runs where both are > 0 (1 = in proportion; NaN with
                         fewer than 2 such runs)
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.fluid import Settings, stats, step
from physics.scenes.blob_start import blob_start
from physics.scenes.sealed_region import sealed_region


def black_hole(side: int = 48, energies: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0), radius: float = 6.0,
               steps: int = 40, settings: Settings | None = None, device: str = "cuda") -> dict:
    """Sealed region and collapsed energy after each ball's fall, and how the radius scales."""
    settings = settings or Settings.all_on()
    middle = (side - 1) / 2.0
    runs = []
    for energy in energies:
        grid = blob_start(side, [{"centre": (middle,) * 3, "radius": radius, "energy": energy,
                                  "velocity": (0.0, 0.0, 0.0)}], device=device)
        for _ in range(steps):
            step(grid, settings)
        runs.append(sealed_region(grid, settings) | {"max_compression": stats(grid, settings)["max_compression"]})
    result = {"side": side, "radius": radius, "steps": steps, "settings": asdict(settings), "energies": list(energies)}
    for key in ("sealed_boxes", "sealed_radius", "sealed_energy", "collapsed_energy", "max_compression"):
        result[key] = [run[key] for run in runs]
    result["radius_per_energy"] = [run["sealed_radius"] / run["collapsed_energy"] if run["collapsed_energy"] > 0 else 0.0
                                   for run in runs]
    result["radius_energy_power"] = _power(result["collapsed_energy"], result["sealed_radius"])
    return result


def _power(energies: list[float], radii: list[float]) -> float:
    pairs = [(math.log(energy), math.log(radius)) for energy, radius in zip(energies, radii)
             if energy > 0 and radius > 0]
    if len({pair[0] for pair in pairs}) < 2:
        return float("nan")
    slope, _ = np.polyfit([pair[0] for pair in pairs], [pair[1] for pair in pairs], 1)
    return float(slope)
