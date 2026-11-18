"""Orbit scene: extra energy in one volume beside a much more packed one; does it stay together and go round?

Contract: `orbit(cells_per_side=32, distances=(2, 3, 4), big_energy=8.0, small_energy=2.0,
steps=2000, every=5, sideways=SIDEWAYS, seed=1, settings=None, device="cuda",
start=None)` (the defaults are the pass mark's run) runs one placed start per distance (orbit_start): an even grid, the middle
energy volume holding big_energy times an even volume's quanta, the volume `distance`
volumes from it along the first axis holding small_energy times, its fluid set moving
along the second axis by a phase ramp of `sideways` radians a box. Every rule is on.
Beside it runs the twin: the same start without the small volume, step for step. The
small blob is what the start volumes hold above the twin's, outside the big one's own
place and the 26 round it (blob_centre): the even grid moves energy by itself, and that
is not the blob. Per sample: the amount at the small volume and the
extra energy's centre. Returns a dict (no verdict):
  cells_per_side, distances, big_energy, small_energy, steps, sideways, seed, settings (dict)
  runs    one dict per distance:
    time, extra        sample times and the small blob's extra quanta at its own volume
    lost_step          the first sampled step at which that is under half the start's
                       (-1: never in this run)
    radius             distance of the extra energy's centre from the big volume, in
                       volumes, per sample
    turned             turns the centre has gone round the big volume in the plane of
                       the first two axes, summed over the samples up to lost_step
    period_squared_over_radius_cubed   (time per turn) ** 2 / radius ** 3 from that
                       angle, 0 when it did not turn
  known   Kepler: period ** 2 / radius ** 3 is the same at every distance
"""
from __future__ import annotations

from dataclasses import asdict

from physics.engine import Grid, Settings, step
from physics.scenes.orbit.definitions.blob_centre import blob_centre
from physics.scenes.orbit.definitions.sideways_ramp import SIDEWAYS
from physics.scenes.orbit.definitions.went_round import went_round
from physics.scenes.orbit.logic.orbit_start import orbit_start
from physics.scenes.run.definitions.start_settings import start_settings
from physics.scenes.run.logic.sampled_run import sampled_run
from physics.scenes.volumes.logic.volume_field import volume_field


def _blob_now(universe: Grid, twin: Grid, settings: Settings, middle: int, small: tuple) -> tuple:
    """Bring the twin to the universe's step, then read the blob: what the universe holds above the twin."""
    while twin.step < universe.step:
        step(twin, settings)
    return blob_centre(volume_field(universe)[0], volume_field(twin)[0], middle, small)


def _one_distance(cells_per_side: int, distance: int, big_energy: float, small_energy: float, steps: int,
                  every: int, sideways: float, seed: int, settings: Settings, device: str) -> dict:
    """One placed run: the blob's amount and centre per sample, and what went_round reads from them."""
    universe, twin, small, middle = orbit_start(cells_per_side, distance, big_energy, small_energy, sideways,
                                                seed, settings, device)
    samples = sampled_run(universe, settings, steps, every,
                          lambda sampled: _blob_now(sampled, twin, settings, middle, small))
    times = [time for time, _ in samples]
    extra = [measured[0] for _, measured in samples]
    found = went_round(times, extra, [measured[1] for _, measured in samples])
    lost = found.pop("lost_sample")
    return {"time": times, "extra": extra,
            "lost_step": -1 if lost < 0 else int(round(times[lost] / settings.step_time)), **found}


def orbit(cells_per_side: int = 32, distances: tuple = (2, 3, 4), big_energy: float = 8.0, small_energy: float = 2.0,
          steps: int = 2000, every: int = 5, sideways: float = SIDEWAYS, seed: int = 1,
          settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """Per distance: how long the small blob keeps its extra energy, and where that energy's centre goes."""
    settings = start_settings(settings, start)
    runs = [_one_distance(cells_per_side, distance, big_energy, small_energy, steps, every, sideways, seed,
                          settings, device) for distance in distances]
    return {"cells_per_side": cells_per_side, "distances": list(distances), "big_energy": big_energy,
            "small_energy": small_energy, "steps": steps, "sideways": sideways, "seed": seed,
            "settings": asdict(settings), "runs": runs}
