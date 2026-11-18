"""Structure scene: how the clustering spectrum of the energy volumes' energy changes over time.

Contract: `structure(cells_per_side=16, seed=1, steps=200, every=50, settings=None,
device="cuda", start=None)` builds physics.engine.new_universe with the settings
(default Settings(); `start` puts a candidate's start amounts in), runs `steps` steps
with every rule on and measures the quanta of every energy volume on the volumes' own
grid (volume_field) at the start and every `every` steps. Returns a dict (no verdict):
  cells_per_side, seed, steps, settings (dict)
  wavenumbers     k = 1 .. volumes a side // 2, cycles per grid side (power_spectrum)
  time            sample times
  power           per sample, the shell power of the energy contrast at each k
  rms_contrast    per sample, the RMS of quanta / mean - 1 over the volumes
  power_growth    per k, power at the last sample over power at the start (0 where the
                  start has none)
  lumps, largest_region   per sample (count_lumps: the most lump volumes joined through
                  ways: lumps over regions, not single cells)
  left_out_share  per sample, the share of plus quanta in volumes a tear made, which
                  have no place on the grid and are not in the spectrum
The relations are the lattice's until a tear or a collapse; a volume that fell into
another counts as empty and its quanta where they went. Raises ValueError for a grid
under 8 places a side.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import Grid, Settings, new_universe
from physics.scenes.run.definitions.start_settings import start_settings
from physics.scenes.run.logic.sampled_run import sampled_run
from physics.scenes.structure.definitions.rms_contrast import rms_contrast
from physics.scenes.volumes.definitions.power_spectrum import power_spectrum
from physics.scenes.volumes.logic.count_lumps import count_lumps
from physics.scenes.volumes.logic.volume_field import volume_field

_SAMPLED = ("power", "rms_contrast", "lumps", "largest_region", "left_out_share")


def _measure(universe: Grid, settings: Settings) -> dict:
    """One sample: the field's spectrum and contrast, the lumps, and what the field misses."""
    field, left_out = volume_field(universe)
    wavenumbers, power = power_spectrum(field)
    return {"wavenumbers": wavenumbers.tolist(), "power": power.tolist(), "rms_contrast": rms_contrast(field),
            **count_lumps(universe, settings), "left_out_share": left_out / max(float(field.sum()) + left_out, 1.0)}


def structure(cells_per_side: int = 16, seed: int = 1, steps: int = 200, every: int = 50,
              settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """The energy contrast's spectrum, RMS and lump counts at each sample, and the spectrum's growth."""
    if cells_per_side < 8:
        raise ValueError(f"a spectrum needs 4 volumes a side, so 8 places or more, got {cells_per_side}")
    settings = start_settings(settings, start)
    universe = new_universe(settings, cells_per_side, seed, device=device)
    samples = sampled_run(universe, settings, steps, every, lambda sampled: _measure(sampled, settings))
    first, last = np.asarray(samples[0][1]["power"]), np.asarray(samples[-1][1]["power"])
    return {"cells_per_side": cells_per_side, "seed": seed, "steps": steps, "settings": asdict(settings),
            "wavenumbers": samples[0][1]["wavenumbers"], "time": [time for time, _ in samples],
            **{name: [measured[name] for _, measured in samples] for name in _SAMPLED},
            "power_growth": np.divide(last, first, out=np.zeros_like(last), where=first > 0.0).tolist()}
