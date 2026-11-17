"""Structure scene: how the clustering spectrum of the energy volumes' energy changes over time.

Contract: `structure(cells_per_side=16, seed=1, unevenness=0.1, steps=200, every=50,
settings=None, device="cuda")` builds physics.engine.new_universe(cells_per_side, seed,
unevenness), runs `steps` steps with settings (default Settings(), every rule on) and
measures the quanta of every energy volume on the volumes' own grid (volume_field.py)
at the start and every `every` steps. Returns a dict (no verdict):
  cells_per_side, seed, unevenness, steps, settings (dict)
  wavenumbers     k = 1 .. volumes a side // 2, cycles per grid side (power_spectrum.py)
  time            sample times
  power           per sample, the shell power of the energy contrast at each k
  rms_contrast    per sample, the RMS of quanta / mean - 1 over the volumes
  power_growth    per k, power at the last sample over power at the start
  lumps, largest_region   per sample, stats "lumps" and "largest_region" (the most
                  lump volumes joined through ways: lumps over regions, not single cells)
  left_out_share  per sample, the share of plus quanta in volumes a tear made, which
                  have no place on the grid and are not in the spectrum
The relations are the lattice's until a tear or a collapse; a volume that fell into
another counts as empty and its quanta where they went.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import Settings, State, new_universe, stats
from physics.scenes.power_spectrum import power_spectrum
from physics.scenes.sampled_run import sampled_run
from physics.scenes.volume_field import volume_field


def structure(cells_per_side: int = 16, seed: int = 1, unevenness: float = 0.1, steps: int = 200, every: int = 50,
              settings: Settings | None = None, device: str = "cuda") -> dict:
    """The energy contrast's spectrum, RMS and lump counts at each sample, and the spectrum's growth."""
    settings = settings or Settings()
    universe = new_universe(cells_per_side, seed, unevenness, device=device)
    samples = sampled_run(universe, settings, steps, every, lambda sampled: _measure(sampled, settings))
    first, last = np.asarray(samples[0][1]["power"]), np.asarray(samples[-1][1]["power"])
    names = ("power", "rms_contrast", "lumps", "largest_region", "left_out_share")
    return {"cells_per_side": cells_per_side, "seed": seed, "unevenness": unevenness, "steps": steps,
            "settings": asdict(settings), "wavenumbers": samples[0][1]["wavenumbers"],
            "time": [time for time, _ in samples],
            **{name: [measured[name] for _, measured in samples] for name in names},
            "power_growth": (last / first).tolist()}


def _measure(universe: State, settings: Settings) -> dict:
    field, left_out = volume_field(universe)
    wavenumbers, power = power_spectrum(field)
    now = stats(universe, settings)
    return {"wavenumbers": wavenumbers.tolist(), "power": power.tolist(),
            "rms_contrast": float(np.sqrt(np.mean((field / field.mean() - 1.0) ** 2))),
            "lumps": now["lumps"], "largest_region": now["largest_region"],
            "left_out_share": left_out / max(now["plus"], 1)}
