"""Orbit scene: extra energy in one volume beside a much more packed one; does it stay together and go round?

Contract: `orbit(cells_per_side=16, distances=(2, 4), big_energy=8.0, small_energy=2.0,
steps=200, every=5, settings=None, device="cuda")` runs one placed start per distance
(packed_start.py): an even grid, the middle energy volume holding big_energy times the
mean volume's quanta, and the volume `distance` volumes from it along x holding
small_energy times. The engine has no way to give energy a sideways speed (crossing
takes all direction off what crosses, bible "Axiom 7 in the engine"), so the small
blob starts at rest: the scene measures whether it holds together and where its
energy goes. The small blob is the extra energy of the lattice volumes outside the big
one's own place and its six way-neighbours: what each holds above its own quanta at
the start (the small volume's start counted as an even volume's), so at the start it
is the small blob alone; later, energy the big volume loses past its neighbours counts
in too. Per sample: the amount at the small volume, and the extra energy's centre as
its weighted mean lattice place.
Returns a dict (no verdict):
  cells_per_side, distances, big_energy, small_energy, steps, settings (dict)
  runs    one dict per distance:
    time, extra        sample times and the small blob's extra quanta at its own volume
    lost_step          the first sampled step at which that is under half the start's
                       (-1: never in this run)
    radius             distance of the extra energy's centre from the big volume, in
                       volumes, per sample
    turned             angle the centre has gone round the big volume in the x-y plane,
                       in turns, summed over the samples up to lost_step
    period_squared_over_radius_cubed   (time per turn) ** 2 / radius ** 3 from that
                       angle, 0 when it did not turn
  known   Kepler: period ** 2 / radius ** 3 is the same at every distance
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from physics.engine import ENERGY_VOLUME, Settings, State
from physics.scenes.packed_start import packed_start
from physics.scenes.sampled_run import sampled_run
from physics.scenes.volume_field import volume_field


def orbit(cells_per_side: int = 16, distances: tuple = (2, 4), big_energy: float = 8.0, small_energy: float = 2.0,
          steps: int = 200, every: int = 5, settings: Settings | None = None, device: str = "cuda") -> dict:
    """Per distance: how long the small blob keeps its extra energy, and where that energy's centre goes."""
    settings = settings or Settings()
    across = cells_per_side // 2
    middle = across // 2
    runs = []
    for distance in distances:
        small = ((middle + distance) % across, middle, middle)
        universe = packed_start(cells_per_side, {(2 * middle,) * 3: big_energy,
                                                 tuple(2 * index for index in small): small_energy},
                                settings=settings, device=device)
        before, _ = volume_field(universe)
        before[small] = universe.start_quanta[ENERGY_VOLUME]
        samples = sampled_run(universe, settings, steps, every,
                              lambda sampled: _blob(sampled, before, middle, small))
        runs.append(_summary(samples, every, settings.step_time))
    return {"cells_per_side": cells_per_side, "distances": list(distances), "big_energy": big_energy,
            "small_energy": small_energy, "steps": steps, "settings": asdict(settings), "runs": runs}


def _blob(universe: State, before: np.ndarray, middle: int, small: tuple) -> tuple:
    """The extra quanta at the small blob's own volume, and the extra energy's centre relative to the big volume."""
    field, _ = volume_field(universe)
    across = field.shape[0]
    extra = np.maximum(field - before, 0.0)
    offset = (np.indices(field.shape) - middle + across // 2) % across - across // 2
    extra[np.abs(offset).sum(axis=0) <= 1] = 0.0
    weight = max(float(extra.sum()), 1.0)
    centre = [float((extra * offset[axis]).sum() / weight) for axis in range(3)]
    return float(extra[small]), centre


def _summary(samples: list, every: int, step_time: float) -> dict:
    """Extra energy and radius per sample, when the blob was lost, and how far its centre went round."""
    extra = [measured[0] for _, measured in samples]
    centres = np.array([measured[1] for _, measured in samples])
    lost = next((index for index, amount in enumerate(extra) if amount < 0.5 * extra[0]), -1)
    kept = centres if lost < 0 else centres[:lost + 1]
    angle = np.unwrap(np.arctan2(kept[:, 1], kept[:, 0]))
    turned = float(angle[-1] - angle[0]) / (2.0 * math.pi)
    radius = np.linalg.norm(centres, axis=1)
    took = samples[len(kept) - 1][0]
    kepler = 0.0
    if abs(turned) > 1e-9 and radius[0] > 0.0:
        kepler = float((took / abs(turned)) ** 2 / radius[0] ** 3)
    return {"time": [time for time, _ in samples], "extra": extra,
            "lost_step": -1 if lost < 0 else int(round(samples[lost][0] / step_time)), "radius": radius.tolist(),
            "turned": turned, "period_squared_over_radius_cubed": kepler}
