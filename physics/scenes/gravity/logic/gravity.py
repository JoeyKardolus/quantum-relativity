"""Gravity: phase 1 of the fit as one scene: stand, fall, coast, fade, orbit, as measured numbers.

Contract: `gravity(cells_per_side=24, steps=1200, every=20, seed=1, settings=None,
device="cuda", start=None, quantum=False)` returns one dict of plain numbers, no verdict
(`definitions/pass_mark.py` scores it from outside). Five runs, each a fresh universe:
  stand   the candidate's own uneven universe, nothing placed: `stand_steps`,
          `stand_collapses`, `stand_tears`, `stand_volumes` left of `start_volumes`
  fade    one lump at rest, read at step 0 beside its twin without the lump: the pull's
          rise over two volumes at 2 to 5 volumes from it (`pull_tilts`) beside
          `newton_tilt` in the same wrapping box (`law_tilts`), and their powers
          against distance (`pull_power`, `law_power`); then run: `lone_drift`, the
          furthest its centre strays, in volumes, and `lone_kept`
  fall    two equal lumps: `fall_apart` (their distance per sample, volumes),
          `fall_closed` (the share of the start distance closed by the end, or by the
          sample they came within `TOUCH` volumes), `fall_sped_up` (1 when the second
          half of that closing was faster than the first), `fall_kept` (the least
          either kept of its extra energy at that sample)
  coast   one kicked lump: `coast_travelled` along the kick in volumes,
          `coast_early_speed` and `coast_late_speed` (volumes a step over the first and
          the last quarter of the run), `coast_kept`
  orbit   a small lump beside a big one, kicked sideways by `orbit_kick` from the pull
          as it stands at the small one: `orbit_turns` round the big one,
          `orbit_nearest`, `orbit_furthest` (volumes), `orbit_kept`, `orbit_big_kept`,
          `orbit_kick` (radians a box)
plus `joint_length`, `volume_length` of the even start and every run's `*_collapses`
and `*_tears`. A lump is followed within `RADIUS` volumes. `start` is a candidate's
start amounts (run/start_settings). Every rule the run's layers have is on.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from physics.engine import ENERGY_VOLUME, JOINT, Settings, new_universe, step
from physics.scenes.along_grid import log_slope, volume_pull
from physics.scenes.gravity.definitions.scene_lumps import BOXES_A_VOLUME, scene_lumps
from physics.scenes.gravity.logic.lump_path import lump_path
from physics.scenes.gravity.logic.lumps_start import lumps_start
from physics.scenes.gravity.logic.scene_start import scene_start
from physics.scenes.run.definitions.start_settings import start_settings
from physics.scenes.two_lumps.definitions.newton_tilt import newton_tilt

RADIUS = 1.5
TOUCH = 2.0 * RADIUS
FADE_FROM, FADE_TO = 2, 5


def _volumes(lumps: list) -> list:
    """Each lump's centre in volumes."""
    return [tuple((one - 2.5) / BOXES_A_VOLUME for one in lump.centre) for lump in lumps]


def _kept(amounts: list, sample: int = -1) -> float:
    """What a lump holds at a sample over what it started with."""
    return amounts[sample] / amounts[0] if amounts[0] > 0.0 else 0.0


def _stand(settings, side, steps, seed, device, quantum) -> dict:
    """The candidate's own uneven universe, stepped; what is left of it."""
    universe = new_universe(settings, side, seed, device=device)
    volumes = int((universe.host("kind") == ENERGY_VOLUME).sum())
    done = 0
    while done < steps and universe.collapses == 0 and universe.tears == 0:
        step(universe, settings, quantum=quantum)
        done += 1
    return {"stand_steps": done, "stand_collapses": int(universe.collapses), "stand_tears": int(universe.tears),
            "start_volumes": volumes, "stand_volumes": int((universe.host("kind") == ENERGY_VOLUME).sum())}


def _fade(settings, side, steps, every, seed, device, quantum) -> dict:
    """The pull round one lump at rest beside the law, then the lone lump's own drift."""
    lumps = scene_lumps("lone", side)
    universe = lumps_start(settings, side, lumps, seed, device)
    even = lumps_start(settings, side, [], seed, device)
    kind, length = even.host("kind"), even.host("length")
    centre = [int(round(one)) for one in _volumes(lumps)[0]]
    wide = side // 2
    deeper = (volume_pull(universe) - volume_pull(even))[:, centre[1], centre[2]]
    tilts = [float(deeper[(centre[0] + far + 1) % wide] - deeper[(centre[0] + far - 1) % wide]) / 2.0
             for far in range(FADE_FROM, FADE_TO + 1)]
    law = [newton_tilt(float(far), 1.0, float(wide)) for far in range(FADE_FROM, FADE_TO + 1)]
    far = list(range(FADE_FROM, FADE_TO + 1))
    path = lump_path(universe, settings, _volumes(lumps), RADIUS, steps, every, quantum)
    start = np.array(path["centres"][0][0])
    return {"pull_tilts": tilts, "law_tilts": law, "pull_power": log_slope(far, tilts), "law_power": log_slope(far, law),
            "lone_drift": float(max(np.linalg.norm(np.array(place) - start) for place in path["centres"][0])),
            "lone_kept": _kept(path["amounts"][0]), "lone_collapses": path["collapses"], "lone_tears": path["tears"],
            "joint_length": float(length[kind == JOINT].mean()), "volume_length": float(length[kind == ENERGY_VOLUME].mean())}


def _fall(settings, side, steps, every, seed, device, quantum) -> dict:
    """Two equal lumps: how much of their distance they close, whether faster and faster, what each keeps."""
    lumps = scene_lumps("two_lumps", side)
    universe = lumps_start(settings, side, lumps, seed, device)
    path = lump_path(universe, settings, _volumes(lumps), RADIUS, steps, every, quantum)
    apart = [float(np.linalg.norm(np.array(second) - np.array(first)))
             for first, second in zip(path["centres"][0], path["centres"][1])]
    last = next((sample for sample, distance in enumerate(apart) if distance <= TOUCH), len(apart) - 1)
    middle = last // 2
    return {"fall_apart": apart, "fall_closed": (apart[0] - apart[last]) / apart[0], "fall_samples": last,
            "fall_sped_up": 1.0 if last >= 2 and apart[middle] - apart[last] > apart[0] - apart[middle] > 0.0 else 0.0,
            "fall_kept": min(_kept(path["amounts"][0], last), _kept(path["amounts"][1], last)),
            "fall_collapses": path["collapses"], "fall_tears": path["tears"]}


def _coast(settings, side, steps, every, seed, device, quantum) -> dict:
    """One kicked lump: how far it goes and what is left of its speed and of itself."""
    lumps = scene_lumps("coast", side)
    universe = lumps_start(settings, side, lumps, seed, device)
    path = lump_path(universe, settings, _volumes(lumps), RADIUS, steps, every, quantum)
    along = [place[0] - path["centres"][0][0][0] for place in path["centres"][0]]
    quarter = max((len(along) - 1) // 4, 1)
    span = quarter * every
    return {"coast_travelled": along[-1], "coast_early_speed": (along[quarter] - along[0]) / span,
            "coast_late_speed": (along[-1] - along[-1 - quarter]) / span, "coast_kept": _kept(path["amounts"][0]),
            "coast_collapses": path["collapses"], "coast_tears": path["tears"]}


def _orbit(settings, side, steps, every, seed, device, quantum) -> dict:
    """A small lump kicked sideways beside a big one: how far round it gets."""
    universe, lumps = scene_start("orbit", settings, side, seed, device)
    path = lump_path(universe, settings, _volumes(lumps), RADIUS, steps, every, quantum)
    arm = np.array(path["centres"][1]) - np.array(path["centres"][0])
    angle = np.unwrap(np.arctan2(arm[:, 1], arm[:, 0]))
    radius = np.hypot(arm[:, 0], arm[:, 1])
    return {"orbit_turns": float(abs(angle[-1] - angle[0]) / (2.0 * math.pi)), "orbit_nearest": float(radius.min()),
            "orbit_furthest": float(radius.max()), "orbit_kept": _kept(path["amounts"][1]),
            "orbit_big_kept": _kept(path["amounts"][0]), "orbit_kick": float(lumps[1].kick[1]),
            "orbit_collapses": path["collapses"], "orbit_tears": path["tears"]}


def gravity(cells_per_side: int = 24, steps: int = 1200, every: int = 20, seed: int = 1,
            settings: Settings | None = None, device: str = "cuda", start: dict | None = None,
            quantum: bool = False) -> dict:
    """Phase 1 measured: does the universe stand, do lumps fall, coast, pull by the law and go round."""
    settings = start_settings(settings, start)
    even = replace(settings, unevenness=0.0)
    side = cells_per_side
    numbers = {"side": side, "steps": steps, "seed": seed}
    numbers.update(_stand(settings, side, steps, seed, device, quantum))
    for part in (_fade, _fall, _coast, _orbit):
        numbers.update(part(even, side, steps, every, seed, device, quantum))
    return numbers
