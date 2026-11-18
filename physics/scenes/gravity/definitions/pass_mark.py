"""Pass mark and score of the gravity scene, written on 2026-11-18 before any search or scored run.

Contract: `gravity_score(numbers)` returns `{"stands", "falls", "coasts", "fades", "orbits",
"score"}`, each from 0 to 1, read off the dict `gravity` returns; `passes(numbers)` is
True when every part is at least `PART_PASSES`. The parts are the owner's phase 1:
  stands  1 when the uneven universe had no collapse and no tear in its run, else 0
  falls   two lumps close in as bodies: `closed / FALLS_FULL` (the share of their start
          distance they closed, at most 1), times the least either kept of its energy
          (`kept`, at most 1, counted 0 below `KEEPS`), halved when they did not speed
          up, times `1 - drift / closed` (never below 0): a lone lump's own drift is
          not a fall
  coasts  a kicked lump keeps going: `travelled / COASTS_FULL` volumes (at most 1),
          times `late_speed / early_speed` (at most 1: what friction and the grid left
          of its speed), times what it kept (0 below `KEEPS`)
  fades   the pull's fall with distance beside the law's in the same wrapping box:
          `1 - abs(power - law_power) / FADES_WITHIN`, never below 0
  orbits  `turns` of the small lump round the big one (at most 1), 0 when it kept less
          than `KEEPS` or the big one did
  score   `stands` times the mean of the other four: 0 for a universe that collapses, 0
          for one in which nothing moves (nothing closes, travels or turns; `fades`
          alone is then dropped too)
A number that is missing or not finite counts 0.
"""
from __future__ import annotations

import math

PART_PASSES = 0.5
KEEPS = 0.5
FALLS_FULL = 0.25
COASTS_FULL = 4.0
FADES_WITHIN = 0.5


def _number(numbers: dict, name: str) -> float:
    """A finite reading, or 0."""
    value = numbers.get(name, 0.0)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def _kept(share: float) -> float:
    """What a lump kept, at most 1, and 0 when it kept less than `KEEPS`."""
    return min(share, 1.0) if share >= KEEPS else 0.0


def gravity_score(numbers: dict) -> dict:
    """The five parts and the score of one `gravity` result."""
    read = lambda name: _number(numbers, name)
    stands = 1.0 if read("stand_collapses") == 0.0 and read("stand_tears") == 0.0 and read("stand_steps") > 0.0 else 0.0
    closed = max(read("fall_closed"), 0.0)
    falls = min(closed / FALLS_FULL, 1.0) * _kept(read("fall_kept"))
    falls *= 1.0 if read("fall_sped_up") > 0.0 else 0.5
    falls *= max(1.0 - read("lone_drift") / closed, 0.0) if closed > 0.0 else 0.0
    early = read("coast_early_speed")
    coasts = min(max(read("coast_travelled"), 0.0) / COASTS_FULL, 1.0) * _kept(read("coast_kept"))
    coasts *= min(max(read("coast_late_speed") / early, 0.0), 1.0) if early > 0.0 else 0.0
    fades = max(1.0 - abs(read("pull_power") - read("law_power")) / FADES_WITHIN, 0.0) if "pull_power" in numbers else 0.0
    orbits = min(max(read("orbit_turns"), 0.0), 1.0) if min(read("orbit_kept"), read("orbit_big_kept")) >= KEEPS else 0.0
    moves = falls + coasts + orbits
    score = stands * (falls + coasts + fades + orbits) / 4.0 if moves > 0.0 else 0.0
    return {"stands": stands, "falls": falls, "coasts": coasts, "fades": fades, "orbits": orbits, "score": score}


def passes(numbers: dict) -> bool:
    """Whether every part of phase 1 holds."""
    parts = gravity_score(numbers)
    return all(parts[name] >= PART_PASSES for name in ("stands", "falls", "coasts", "fades", "orbits"))
