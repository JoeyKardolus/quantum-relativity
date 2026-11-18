"""Black hole scene: how far round a packed core does nothing get away, and does that distance grow with the core's energy?

A horizon is not the core: it is the affected region round it (owner, 2026-11-17). The
measure, fixed before any run and built only from what the rules already give:
  held region   the volumes round the core through whose boundaries net plus energy
                flows inward on every way out (held_shells, shell_ways)
  held radius   the summed lengths of the cells from the core's middle to the middle of
                the region's outermost volume, along the six rows through the core
  depth radius  the second reading: the distance at which the pull's depth falls to a
                fixed level, the even twin's own pull scatter along the same rows
Known answer: both radii in proportion to the core's energy (power +1).

Contract: `black_hole(cells_per_side=24, energies=(2, 3, 4, 6, 8, 12, 16, 24, 32),
steps=20000, window=16000, every=20, seed=1, settings=None, device="cuda", start=None)`
runs one placed start per energy (packed_start, which `start` reaches: an even grid
whose middle volume holds `energy` times an even volume's quanta) for `steps` steps
with every rule on, and once the same start with no core (the twin, subtracted from
every reading). Readings are taken over the last `window` steps, the pull every `every`
steps (core_run). Returns a dict (numbers and lists, no verdict):
  cells_per_side, steps, window, every, seed, settings (dict), energies, known_power (1)
  shells        boundaries read, 0 .. a quarter of the cells a side; the held region
                can reach one less (held_limit), the depth radius the last (depth_limit)
  level         the twin's pull scatter (core_run `scatter`)
  per energy, a list each: everything core_numbers returns but `far` and `box`
  held_power, depth_power   least-squares slope of ln radius against ln core_energy
                over the runs whose radius is above 0 and under its limit (NaN with
                fewer than 2)
  depth_law_power   beside depth_power: the slope a pull of one over distance from a
                source in proportion to energy gives at the same radii in the same
                wrapping box (box_depth: the core's copies pull too)
Shells are the start lattice's; a tear or collapse near the core is counted, not
followed. Raises ValueError unless 0 < window <= steps, the grid has 8 places a side or
more and there is at least one energy.
"""
from __future__ import annotations

from dataclasses import asdict

from physics.engine import Settings
from physics.scenes.along_grid.definitions.box_depth import box_depth
from physics.scenes.along_grid.definitions.log_slope import log_slope
from physics.scenes.black_hole.logic.core_numbers import core_numbers
from physics.scenes.black_hole.logic.core_run import core_run
from physics.scenes.placed_start.logic.packed_start import packed_start
from physics.scenes.run.definitions.start_settings import start_settings

_ASIDE = ("far", "box")


def black_hole(cells_per_side: int = 24, energies: tuple = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0),
               steps: int = 20000, window: int = 16000, every: int = 20, seed: int = 1,
               settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """Per packed energy: how far round the core energy flows inward, and how far the pull's depth reaches."""
    if not 0 < window <= steps or cells_per_side < 8 or not energies:
        raise ValueError("window must be 1 .. steps, the grid 8 places a side or more, and one energy or more given")
    settings = start_settings(settings, start)
    middle = cells_per_side // 2 - (cells_per_side // 2) % 2
    twin = core_run(packed_start(cells_per_side, {}, seed, settings, device), middle, steps, window, every, settings)
    runs = []
    for energy in energies:
        universe = packed_start(cells_per_side, {(middle,) * 3: energy}, seed, settings, device)
        read = core_run(universe, middle, steps, window, every, settings)
        runs.append(core_numbers(universe, read, twin, twin["scatter"], settings))
    held = [run for run in runs if run["held_radius"] > 0.0 and not run["held_limit"]]
    deep = [run for run in runs if run["depth_radius"] > 0.0 and not run["depth_limit"]]
    result = {"cells_per_side": cells_per_side, "steps": steps, "window": window, "every": every, "seed": seed,
              "settings": asdict(settings), "energies": list(energies), "known_power": 1.0,
              "shells": len(twin["rows"]) - 1, "level": twin["scatter"]}
    result.update({name: [run[name] for run in runs] for name in runs[0] if name not in _ASIDE})
    result["held_power"] = log_slope([run["core_energy"] for run in held], [run["held_radius"] for run in held])
    result["depth_power"] = log_slope([run["core_energy"] for run in deep], [run["depth_radius"] for run in deep])
    result["depth_law_power"] = log_slope([1.0 / box_depth(run["depth_radius"], run["far"], run["box"]) for run in deep],
                                          [run["depth_radius"] for run in deep])
    return result
