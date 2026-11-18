"""Isotropy scene: are the pull and the flow round one lump the same along the grid's axes and off them?

Contract: `isotropy(cells_per_side=48, factor=4.0, steps=2000, settle=1000, pairs=PAIRS,
diagonals=DIAGONALS, seed=1, settings=None, start=None, device="cuda")` steps two placed
starts side by side (packed_start): an even grid with its middle energy volume packed
`factor` times, and the same grid with nothing packed (the same seed, so the same jitter
inside the boxes), `steps` steps each with every rule on. What the lump adds is read as
the difference between the two, at start volumes counted from the lump in volumes along
the three lattice axes; every mirror and swap of a place is averaged (mirror_mean).
Returns a dict (numbers and lists, no verdict):
  cells_per_side, factor, steps, settle, seed, settings (dict)
  kept           the share of the lump's extra energy still in its own volume at the end
  pairs          one dict per pair of places at the same straight-line distance, the
                 first on an axis and the second off it:
    on, off, distance   the two places and their straight-line distance in volumes
    pull_on, pull_off   how much deeper the pull lies there than at the place farthest
                        from the lump, the mean over the steps after `settle`
    pull_ratio          pull_off / pull_on (0.0 when pull_on is 0)
    law_pull_ratio      the same ratio for a 1 / distance pull from a point in the same
                        wrapping box (box_field): the box is a cube, so even a law with
                        no preferred direction is not exactly 1
    flow_on, flow_off   quanta there above the even twin's at the end: what arrived
    flow_ratio          flow_off / flow_on; plain spreading with no preferred direction
                        gives 1 (law_flow_ratio)
  diagonals      the same numbers at `diagonals`: pull and flow each as a ratio to the
                 first place's (no law beside them: their straight-line distances differ)
Raises ValueError for a place a quarter of the cells a side away or more, or steps < 0.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from physics.engine import Settings, step
from physics.scenes.along_grid.definitions.box_field import box_field
from physics.scenes.along_grid.logic.volume_pull import volume_pull
from physics.scenes.isotropy.definitions.mirror_mean import mirror_mean
from physics.scenes.isotropy.definitions.places import DIAGONALS, PAIRS
from physics.scenes.placed_start.logic.packed_start import packed_start
from physics.scenes.run.definitions.even_volume_quanta import even_volume_quanta
from physics.scenes.run.definitions.start_settings import start_settings
from physics.scenes.volumes.logic.volume_field import volume_field


def _ratio(above: float, below: float) -> float:
    """above / below, 0.0 when below is 0."""
    return float(above / below) if below != 0.0 else 0.0


def _pair_numbers(read: dict, on: tuple, off: tuple) -> dict:
    """One pair's pull, law and flow, on the axis and off it."""
    return {"on": list(on), "off": list(off), "distance": float(np.linalg.norm(on)),
            "pull_on": read[on][0], "pull_off": read[off][0], "pull_ratio": _ratio(read[off][0], read[on][0]),
            "law_pull_ratio": _ratio(read[off][2], read[on][2]),
            "flow_on": read[on][1], "flow_off": read[off][1], "flow_ratio": _ratio(read[off][1], read[on][1]),
            "law_flow_ratio": 1.0}


def isotropy(cells_per_side: int = 48, factor: float = 4.0, steps: int = 2000, settle: int = 1000,
             pairs: tuple = PAIRS, diagonals: tuple = DIAGONALS, seed: int = 1, settings: Settings | None = None,
             start: dict | None = None, device: str = "cuda") -> dict:
    """Pull and arrived energy round one placed lump, on an axis against off it at the same distance."""
    across = cells_per_side // 2
    places = [place for pair in pairs for place in pair] + list(diagonals)
    if steps < 0 or max(max(place) for place in places) >= across // 2:
        raise ValueError(f"need steps >= 0 and places under {across // 2} volumes from the lump, got {places}")
    settings = start_settings(settings, start)
    middle = (cells_per_side // 4) * 2
    lump = packed_start(cells_per_side, {(middle,) * 3: factor}, seed, settings, device)
    even = packed_start(cells_per_side, {}, seed, settings, device)
    depth = np.zeros((across,) * 3)
    for taken in range(1, steps + 1):
        step(lump, settings)
        step(even, settings)
        if taken > settle:
            depth += volume_pull(even) - volume_pull(lump)
    depth /= max(steps - settle, 1)
    centre = middle // 2
    far = (centre + across // 2) % across
    depth = depth - depth[far, far, far]
    arrived = volume_field(lump)[0] - volume_field(even)[0]
    law = box_field(across)
    read = {place: (mirror_mean(depth, centre, place), mirror_mean(arrived, centre, place), mirror_mean(law, 0, place))
            for place in places}
    extra = (factor - 1.0) * even_volume_quanta(settings)
    return {"cells_per_side": cells_per_side, "factor": factor, "steps": steps, "settle": settle, "seed": seed,
            "settings": asdict(settings), "kept": float(arrived[centre, centre, centre] / extra),
            "pairs": [_pair_numbers(read, on, off) for on, off in pairs],
            "diagonals": [{"place": list(place), "pull_ratio": _ratio(read[place][0], read[diagonals[0]][0]),
                           "flow_ratio": _ratio(read[place][1], read[diagonals[0]][1])} for place in diagonals]}
