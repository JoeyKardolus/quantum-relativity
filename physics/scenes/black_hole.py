"""Black hole scene: does a packed volume seal itself, and does the sealed region grow with its energy?

Contract: `black_hole(cells_per_side=12, energies=(4.0, 8.0, 16.0), steps=100,
settings=None, device="cuda")` runs one placed start per energy (packed_start.py): an
even grid whose middle energy volume holds `energy` times the mean volume's quanta.
After `steps` steps with settings (default Settings(), every rule on) it reads the
sealed region round the most packed volume (sealed_region.py). Returns a dict (no
verdict):
  cells_per_side, steps, settings (dict), energies
  sealed, sealed_volumes, sealed_radius, sealed_energy, core_quanta, core_joints,
  core_sealed, collapses, merges, squeeze_max   one per energy, at the end
  radius_per_energy      sealed_radius / sealed_energy per run (0 when not sealed);
                         relativity's horizon keeps this constant
  radius_energy_power    least-squares slope of ln sealed_radius against ln sealed_energy
                         over the runs that sealed (1 = in proportion; NaN with fewer
                         than 2 such runs)
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.engine import Settings, stats, step
from physics.scenes.packed_start import packed_start
from physics.scenes.sealed_region import sealed_region


def black_hole(cells_per_side: int = 12, energies: tuple = (4.0, 8.0, 16.0), steps: int = 100,
               settings: Settings | None = None, device: str = "cuda") -> dict:
    """Per packed energy: whether the region round it sealed, its radius and energy."""
    settings = settings or Settings()
    middle = cells_per_side // 2 - (cells_per_side // 2) % 2
    runs = []
    for energy in energies:
        universe = packed_start(cells_per_side, {(middle, middle, middle): energy}, settings=settings, device=device)
        for _ in range(steps):
            step(universe, settings)
        final = stats(universe, settings)
        runs.append({**sealed_region(universe, settings), "collapses": final["collapses"],
                     "merges": final["merges"], "squeeze_max": final["squeeze_max"]})
    closed = [run for run in runs if run["sealed"] and run["radius"] > 0.0 and run["energy"] > 0]
    power = float("nan")
    if len(closed) >= 2:
        power = float(np.polyfit([math.log(run["energy"]) for run in closed],
                                 [math.log(run["radius"]) for run in closed], 1)[0])
    return {"cells_per_side": cells_per_side, "steps": steps, "settings": asdict(settings),
            "energies": list(energies), "sealed": [run["sealed"] for run in runs],
            "sealed_volumes": [run["volumes"] for run in runs], "sealed_radius": [run["radius"] for run in runs],
            "sealed_energy": [run["energy"] for run in runs], "core_quanta": [run["core_quanta"] for run in runs],
            "core_joints": [run["core_joints"] for run in runs], "core_sealed": [run["core_sealed"] for run in runs],
            "collapses": [run["collapses"] for run in runs], "merges": [run["merges"] for run in runs],
            "squeeze_max": [run["squeeze_max"] for run in runs],
            "radius_per_energy": [run["radius"] / run["energy"] if run["energy"] else 0.0 for run in runs],
            "radius_energy_power": power}
