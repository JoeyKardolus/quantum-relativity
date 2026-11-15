"""Structure scene: how the clustering spectrum of the density field changes over time.

Contract: `structure(cells_per_side=8, boxes_per_cell_side=6, seed=1, unevenness=0.3,
steps=30, every=5, settings=None, device="cuda")` builds
physics.potts.new_universe(cells_per_side, boxes_per_cell_side, seed, unevenness), runs
`steps` steps with settings (default Settings.all_on()) and measures the energy per
Planck box (A plus B, box_energies.py) at the start and every `every` steps. Returns a
dict (no verdict):
  cells_per_side, boxes_per_cell_side, seed, unevenness, steps, settings (dict)
  wavenumbers     k = 1 .. side // 2, cycles per grid side (power_spectrum.py)
  time            sample times
  power           per sample, the shell power of the density contrast at each k
  rms_contrast    per sample, the RMS of density / mean - 1 over the grid
  power_growth    per k, power at the last sample over power at the start
  lumps           per sample, stats "lumps"
The whole grid is transformed, the empty margin included (power_spectrum.py), so the
checkerboard of cells and the filled cube's edge add power of their own.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.potts import Settings, Universe, new_universe, stats
from physics.scenes.box_energies import box_energies
from physics.scenes.power_spectrum import power_spectrum
from physics.scenes.sampled_run import sampled_run


def structure(cells_per_side: int = 8, boxes_per_cell_side: int = 6, seed: int = 1, unevenness: float = 0.3,
              steps: int = 30, every: int = 5, settings: Settings | None = None, device: str = "cuda") -> dict:
    """The density contrast's spectrum, RMS and lump count at each sample, and the spectrum's growth."""
    settings = settings or Settings.all_on()
    universe = new_universe(cells_per_side, boxes_per_cell_side, seed, unevenness, device=device)
    samples = sampled_run(universe, settings, steps, every, lambda sampled: _measure(sampled, settings))
    first, last = np.asarray(samples[0][1]["power"]), np.asarray(samples[-1][1]["power"])
    return {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "seed": seed,
            "unevenness": unevenness, "steps": steps, "settings": asdict(settings),
            "wavenumbers": samples[0][1]["wavenumbers"], "time": [time for time, _ in samples],
            "power": [measured["power"] for _, measured in samples],
            "rms_contrast": [measured["rms_contrast"] for _, measured in samples],
            "power_growth": (last / first).tolist(), "lumps": [measured["lumps"] for _, measured in samples]}


def _measure(universe: Universe, settings: Settings) -> dict:
    a_energy, b_energy = box_energies(universe)
    density = a_energy + b_energy
    wavenumbers, power = power_spectrum(density)
    return {"wavenumbers": wavenumbers.tolist(), "power": power.tolist(),
            "rms_contrast": float(np.sqrt(np.mean((density / density.mean() - 1.0) ** 2))),
            "lumps": stats(universe, settings)["lumps"]}
