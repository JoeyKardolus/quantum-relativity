"""Black hole scene: how far round a packed core does nothing get away, and does that distance grow with the core's energy?

A horizon is not the core: it is the affected region round it (owner, 2026-11-17). The
measure, fixed before any run and built only from what the rules already give (the pull
that spread from the core at its finite speed, and the crossing that pull steers):
  held region   the volumes round the core through whose boundaries net plus energy
                flows inward on every way out. Volumes are sorted in shells by how many
                ways lie between them and the core; boundary j closes the core and the
                shells 1 .. j. The region reaches as far as every boundary from 1 on
                took energy in over the reading window (energy placed there ends up
                nearer the core, it does not leave). The core's own skin, boundary 0,
                is left out and reported apart (core_gain): nothing is assumed about
                the core
  held radius   the summed lengths of the cells from the core's middle to the middle of
                the region's outermost volume, along the six rows through the core
  depth radius  the second reading: the distance at which the pull's depth falls to a
                fixed level, the even grid's own heat per quantum (where the pull bends
                a crossing by the factor e, cross.py). Depth at a shell is how much
                lower the pull lies there than at the row's volume half the box away
Known answer: both radii in proportion to the core's energy (power +1).

Contract: `black_hole(cells_per_side=24, energies=(2, 3, 4, 6, 8, 12, 16, 24, 32),
steps=20000, window=16000, every=20, seed=1, settings=None, device="cuda", start=None)`
runs one placed start per energy (packed_start, which `start` reaches: an even grid
whose middle volume holds `energy` times the mean volume's quanta) for `steps` steps
with every rule on, and once the same start with no core (the twin, subtracted from
every reading: the even grid's own jitter moves energy and tilts the pull). Readings
are taken over the last `window` steps, the pull every `every` steps: it is a wave that
keeps swinging, and a core passes it on at its own clock's rate (pull_wave.py), so a
more packed core's pull arrives later; core_clock says how much time the core had. Returns a dict (numbers and lists, no verdict):
  cells_per_side, steps, window, every, seed, settings (dict), energies, known_power (1)
  shells        boundaries read, 0 .. a quarter of the volumes a side; the held region
                can reach one less (held_limit), the depth radius the last (depth_limit)
  level         the twin's mean heat per quantum over the window
  per energy, at the end:
  core_energy   the core's quanta in mean start volumes; core_clock its local time;
                core_joints, core_sealed (sealed_region.py); core_gain the energy that
                went into the core over the window, in mean start volumes
  distance      per shell, summed lengths from the core's middle to that shell's
  inflow        per boundary, the energy that crossed it inward over the window, in
                mean start volumes, twin subtracted; gathered the same since the start
  depth         per shell, the pull's depth below the half-box volume, window mean
  held_volumes, held_radius, held_limit (1: the region reached the last shell read)
  depth_radius, depth_limit (1: the depth is above the level out to the last shell)
  collapses, merges, tears, squeeze_max
  held_power, depth_power   least-squares slope of ln radius against ln core_energy
                over the runs whose radius is above 0 and under its limit (NaN with
                fewer than 2)
  depth_law_power   beside depth_power: the slope a pull of one over distance from a
                source in proportion to energy gives at the same radii in the same
                wrapping box (the core's copies pull too)
Shells are the start lattice's; a tear or collapse near the core is counted, not followed.
Raises ValueError unless 0 < window <= steps and the grid has 8 places a side or more.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import ENERGY_VOLUME, Settings, State, stats, step
from physics.scenes.packed_start import packed_start
from physics.scenes.sealed_region import sealed_region
from physics.scenes.volume_field import volume_field

_COPIES = 4


def black_hole(cells_per_side: int = 24, energies: tuple = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0),
               steps: int = 20000, window: int = 16000, every: int = 20, seed: int = 1,
               settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """Per packed energy: how far round the core energy flows inward, and how far the pull's depth reaches."""
    settings = settings or Settings()
    if not 0 < window <= steps or cells_per_side < 8:
        raise ValueError("window must be 1 .. steps and the grid 8 places a side or more")
    middle = cells_per_side // 2 - (cells_per_side // 2) % 2
    twin = _read(packed_start(cells_per_side, {}, seed, settings, device, start), middle, steps, window, every, settings)
    level = twin["heat"]
    runs = []
    for energy in energies:
        universe = packed_start(cells_per_side, {(middle, middle, middle): energy}, seed, settings, device, start)
        read = _read(universe, middle, steps, window, every, settings)
        runs.append(_measure(universe, read, twin, level, settings))
    held = [run for run in runs if run["held_radius"] > 0.0 and not run["held_limit"]]
    deep = [run for run in runs if run["depth_radius"] > 0.0 and not run["depth_limit"]]
    result = {"cells_per_side": cells_per_side, "steps": steps, "window": window, "every": every, "seed": seed,
              "settings": asdict(settings), "energies": list(energies), "known_power": 1.0,
              "shells": len(twin["rows"]) - 1, "level": level}
    result.update({name: [run[name] for run in runs] for name in runs[0] if name not in ("box", "far")})
    result["held_power"] = _power([run["core_energy"] for run in held], [run["held_radius"] for run in held])
    result["depth_power"] = _power([run["core_energy"] for run in deep], [run["depth_radius"] for run in deep])
    result["depth_law_power"] = _power([1.0 / _box_law(run["depth_radius"], run["far"], run["box"]) for run in deep],
                                       [run["depth_radius"] for run in deep])
    return result


def _read(universe: State, middle: int, steps: int, window: int, every: int, settings: Settings) -> dict:
    """Step a universe to the end; over the window, the mean pull on the rows, the mean heat and the energy that entered every boundary, over the window and since the start."""
    side = universe.side
    shells = side // 4
    core = (middle * side + middle) * side + middle
    rows = np.stack([_line(universe, core, weight, sign, 2 * shells)[::2]
                     for weight in (side * side, side, 1) for sign in (1, -1)], axis=1)
    volumes = np.indices((side // 2,) * 3) - middle // 2
    ways = np.minimum(np.abs(volumes), side // 2 - np.abs(volumes)).sum(axis=0)
    pulls, heats, inside = [], [], {}
    for taken in range(steps + 1):
        if taken:
            step(universe, settings)
        if taken in (0, steps - window, steps):
            held = volume_field(universe)[0]
            inside[taken] = np.array([held[ways <= shell].sum() for shell in range(shells + 1)])
        if taken > steps - window and (steps - taken) % every == 0:
            pulls.append(universe.host("pull").astype(np.float64)[rows].mean(axis=1))
            heats.append(float(universe.host("jiggle")[universe.kind_h[:universe.count] == ENERGY_VOLUME].mean()))
    return {"rows": rows, "pull": np.mean(pulls, axis=0), "heat": float(np.mean(heats)),
            "entered": inside[steps] - inside[steps - window], "gathered": inside[steps] - inside[0]}


def _measure(universe: State, read: dict, twin: dict, level: float, settings: Settings) -> dict:
    """One run's numbers: the core, the flow through every boundary, the depth at every shell and the two radii."""
    even = universe.start_quanta[ENERGY_VOLUME]
    length = universe.host("length").astype(np.float64)
    rows = read["rows"]
    core = int(rows[0, 0])
    distance = _distances(universe, length, core, len(rows) - 1)
    inflow = (read["entered"] - twin["entered"]) / even
    lower = twin["pull"] - read["pull"]
    depth = lower - lower[-1]
    held = 0
    while held + 1 < len(rows) - 1 and inflow[held + 1] > 0.0:
        held += 1
    deep = next((shell for shell in range(len(rows)) if depth[shell] < level), len(rows) - 1)
    radius = 0.0
    if deep > 0:
        share = (depth[deep - 1] - level) / (depth[deep - 1] - depth[deep])
        radius = float(distance[deep - 1] + share * (distance[deep] - distance[deep - 1]))
    final = stats(universe, settings)
    seal = sealed_region(universe, settings)
    return {"core_energy": float(universe.host("cell_quanta")[core]) / even,
            "core_clock": float(universe.host("clock")[core]), "core_joints": seal["core_joints"],
            "core_sealed": seal["core_sealed"], "core_gain": float(inflow[0]), "distance": distance.tolist(),
            "inflow": inflow.tolist(), "gathered": ((read["gathered"] - twin["gathered"]) / even).tolist(), "depth": depth.tolist(), "held_volumes": held,
            "held_radius": float(distance[held]) if held else 0.0, "held_limit": int(held == len(rows) - 2),
            "depth_radius": radius, "depth_limit": int(deep == len(rows) - 1),
            "collapses": final["collapses"], "merges": final["merges"], "tears": final["tears"],
            "squeeze_max": final["squeeze_max"], "far": float(distance[-1]),
            "box": _round(universe, length, core)}


def _line(universe: State, core: int, weight: int, sign: int, places: int) -> np.ndarray:
    """The ids of the cells from the core along one row, `places` lattice places out (wrapping)."""
    side = universe.side
    index = (core // weight) % side
    return core + (((index + sign * np.arange(places + 1)) % side) - index) * weight


def _distances(universe: State, length: np.ndarray, core: int, shells: int) -> np.ndarray:
    """Per shell, the summed lengths from the core's middle to that shell's volume's middle, the mean of the six rows."""
    side = universe.side
    found = np.zeros(shells + 1)
    for weight in (side * side, side, 1):
        for sign in (1, -1):
            along = length[_line(universe, core, weight, sign, 2 * shells)]
            summed = np.cumsum(along) - 0.5 * along - 0.5 * along[0]
            found += summed[::2] / 6.0
    return found


def _round(universe: State, length: np.ndarray, core: int) -> float:
    """The box: the summed lengths once round a row through the core, the mean of the three rows."""
    side = universe.side
    return float(np.mean([length[_line(universe, core, weight, 1, side - 1)].sum() for weight in (side * side, side, 1)]))


def _box_law(radius: float, far: float, box: float) -> float:
    """How much deeper a 1 / distance pull lies at `radius` than at `far` along a row, from a point and its copies in the neighbouring boxes."""
    shifts = np.arange(-_COPIES, _COPIES + 1) * box
    east, north, up = np.meshgrid(shifts, shifts, shifts, indexing="ij")
    depth = [(1.0 / np.sqrt((at + east) ** 2 + north ** 2 + up ** 2)).sum() for at in (radius, far)]
    return float(depth[0] - depth[1])


def _power(energy: list, radius: list) -> float:
    """Least-squares slope of ln(radius) against ln(energy); NaN with fewer than 2 points."""
    if len(radius) < 2 or min(energy) <= 0.0:
        return float("nan")
    return float(np.polyfit(np.log(energy), np.log(radius), 1)[0])
