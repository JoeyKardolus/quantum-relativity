"""Black hole scene: the sealed region's radius against the energy that fell together.

Contract: `black_hole(cells_per_side=10, boxes_per_cell_side=4, energies=(250.0, 500.0,
1000.0, 2000.0), radius=6.0, steps=40, settings=None, device="cuda")` runs one start
per energy: an even checkerboard holding a ball of that much extra energy at rest in
its middle (blob_start.py, energies in units of 1000 quanta), falling in on itself by its
own pull for `steps` steps with settings (default Settings(), every rule on). After the
run it measures the sealed region round the most packed energy volume (sealed_region.py;
core_labels counts the start's energy volumes it holds). Returns a dict (no verdict):
  cells_per_side, boxes_per_cell_side, radius, steps, settings (dict), energies
  sealed, sealed_boxes, sealed_radius, sealed_energy, collapsed_energy, core_boxes,
  core_labels, core_compression_min   one per energy
  max_compression        stats "squeeze_max" at the end, one per energy
  collapses              stats "collapses" at the end (joint boxes), one per energy
  radius_per_energy      sealed_radius / collapsed_energy per run (0 when not sealed);
                         relativity's horizon keeps this constant
  radius_energy_power    least-squares slope of ln sealed_radius against ln collapsed_energy
                         over the runs where both are > 0 (1 = in proportion; NaN with
                         fewer than 2 such runs)
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.engine import Settings, cells, host_boxes, stats, step
from physics.scenes.blob_start import blob_start
from physics.scenes.sealed_region import sealed_region

_KEYS = ("sealed", "sealed_boxes", "sealed_radius", "sealed_energy", "collapsed_energy", "core_boxes", "core_labels",
         "core_compression_min", "max_compression", "collapses")


def black_hole(cells_per_side: int = 10, boxes_per_cell_side: int = 4,
               energies: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0), radius: float = 6.0, steps: int = 40,
               settings: Settings | None = None, device: str = "cuda") -> dict:
    """Sealed region and fallen-together energy after each ball's fall, and how the radius scales."""
    settings = settings or Settings()
    runs = []
    for energy in energies:
        universe = blob_start(cells_per_side, boxes_per_cell_side, [{"offset": (0.0, 0.0, 0.0), "radius": radius,
                                                                     "energy": energy, "velocity": (0.0, 0.0, 0.0)}],
                              device=device)
        start_volumes = cells(host_boxes(universe))["owner"]
        start_volumes = np.where(host_boxes(universe)["quanta"] > 0, start_volumes, -1)
        for _ in range(steps):
            step(universe, settings)
        end = stats(universe, settings)
        runs.append(sealed_region(universe, settings, start_volumes) | {"max_compression": end["squeeze_max"],
                                                         "collapses": float(end["collapses"])})
    result = {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "radius": radius,
              "steps": steps, "settings": asdict(settings), "energies": list(energies)}
    for key in _KEYS:
        result[key] = [run[key] for run in runs]
    result["radius_per_energy"] = [run["sealed_radius"] / run["collapsed_energy"] if run["sealed"] else 0.0
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
