"""Pass mark of the black hole scene (the affected region), written 2026-11-17 before the official run.

Known: horizon radius in proportion to core energy, power +1.
Contract: `LOWEST` 0.75 and `HIGHEST` 1.25 bound the power; `ENERGIES` is 4 and `SPAN`
8. `passes(result)` takes the scene's result (24 places, energies 2 3 4 6 8 12 16 24 32,
20,000 steps, window the last 16,000) and returns True when at least ENERGIES runs have
a held radius above 0 and below the box's limit, their core energies span a factor
SPAN or more, and the power of held radius against core energy (`held_power`) lies in
LOWEST .. HIGHEST. The flow reading decides the scene; the depth radius is reported
beside it under the same band.
Changed 2026-11-18 with the rebuilt engine, before any run on it: the depth reading's
level is the twin's own pull scatter (heat per quantum no longer exists), and shells
count 26 ways a volume. The band, the count and the span are unchanged.
"""
from __future__ import annotations

LOWEST = 0.75
HIGHEST = 1.25
ENERGIES = 4
SPAN = 8.0


def passes(result: dict) -> bool:
    """Whether the held radius grows in proportion to the core's energy over enough runs."""
    read = [energy for energy, radius, limit in zip(result["core_energy"], result["held_radius"], result["held_limit"])
            if radius > 0.0 and not limit]
    enough = len(read) >= ENERGIES and min(read) > 0.0 and max(read) / min(read) >= SPAN
    return bool(enough and LOWEST <= result["held_power"] <= HIGHEST)
