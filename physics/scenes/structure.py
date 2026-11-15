"""Structure scene: how the clustering spectrum of the density field changes over time.

Contract: `structure(side=64, seed=1, unevenness=0.3, steps=30, every=5, settings=None,
device="cuda")` builds physics.fluid.new_grid(side, seed, unevenness), runs `steps` steps
with settings (default Settings.all_on()) and measures the A- plus B-energy per box at
the start and every `every` steps. Returns a dict (no verdict):
  side, seed, unevenness, steps, settings (dict)
  wavenumbers     k = 1 .. side // 2, cycles per cube side (power_spectrum.py)
  time            sample times
  power           per sample, the shell power of the density contrast at each k
  rms_contrast    per sample, the RMS of density / mean - 1 over the cube
  power_growth    per k, power at the last sample over power at the start
  lumps           per sample, stats "lumps"
The whole cube is transformed, the empty margin included (power_spectrum.py).
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.fluid import Grid, Settings, new_grid, stats
from physics.scenes.box_energies import box_energies
from physics.scenes.power_spectrum import power_spectrum
from physics.scenes.sampled_run import sampled_run


def structure(side: int = 64, seed: int = 1, unevenness: float = 0.3, steps: int = 30, every: int = 5,
              settings: Settings | None = None, device: str = "cuda") -> dict:
    """The density contrast's spectrum, RMS and lump count at each sample, and the spectrum's growth."""
    settings = settings or Settings.all_on()
    grid = new_grid(side, seed, unevenness, device=device)
    samples = sampled_run(grid, settings, steps, every, lambda sampled: _measure(sampled, settings))
    first, last = np.asarray(samples[0][1]["power"]), np.asarray(samples[-1][1]["power"])
    return {"side": side, "seed": seed, "unevenness": unevenness, "steps": steps, "settings": asdict(settings),
            "wavenumbers": samples[0][1]["wavenumbers"], "time": [time for time, _ in samples],
            "power": [measured["power"] for _, measured in samples],
            "rms_contrast": [measured["rms_contrast"] for _, measured in samples],
            "power_growth": (last / first).tolist(), "lumps": [measured["lumps"] for _, measured in samples]}


def _measure(grid: Grid, settings: Settings) -> dict:
    a_energy, b_energy = box_energies(grid)
    density = a_energy + b_energy
    wavenumbers, power = power_spectrum(density)
    return {"wavenumbers": wavenumbers.tolist(), "power": power.tolist(),
            "rms_contrast": float(np.sqrt(np.mean((density / density.mean() - 1.0) ** 2))),
            "lumps": stats(grid, settings)["lumps"]}
