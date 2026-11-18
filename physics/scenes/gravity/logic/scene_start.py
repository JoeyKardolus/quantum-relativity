"""Scene start: the start of one named gravity scene, ready to step or to watch.

Contract: `scene_start(name, settings, side, seed=1, device="cuda")` returns
`(universe, lumps)`: the universe `lumps_start` lays for `scene_lumps(name, side)` and
the lumps as placed. For `orbit` the small lump is first laid at rest, the pull's rise
from box to box is read where it stands (`volume_pull`, one volume either side along
the line to the big lump), and the start is laid again with the sideways kick
`orbit_kick` asks for at that distance: the kick is worked out from the pull, never
picked.
"""
from __future__ import annotations

from physics.engine import Grid, Settings
from physics.scenes.along_grid import volume_pull
from physics.scenes.gravity.definitions.lump import Lump
from physics.scenes.gravity.definitions.orbit_kick import orbit_kick
from physics.scenes.gravity.definitions.scene_lumps import BOXES_A_VOLUME, scene_lumps
from physics.scenes.gravity.logic.lumps_start import lumps_start


def scene_start(name: str, settings: Settings, side: int, seed: int = 1, device: str = "cuda") -> tuple:
    """The universe of one gravity scene and the lumps placed in it."""
    lumps = scene_lumps(name, side)
    universe = lumps_start(settings, side, lumps, seed, device)
    if name != "orbit":
        return universe, lumps
    big, small = lumps
    wide = side // 2
    place = [int(round((one - 2.5) / BOXES_A_VOLUME)) for one in small.centre]
    pull = volume_pull(universe)
    pull_step = float(pull[(place[0] + 1) % wide, place[1], place[2]]
                      - pull[(place[0] - 1) % wide, place[1], place[2]]) / (2.0 * BOXES_A_VOLUME)
    kick = orbit_kick(pull_step, small.centre[0] - big.centre[0], settings.step_time)
    lumps = [big, Lump(small.centre, small.width, small.peak, (0.0, kick, 0.0))]
    return lumps_start(settings, side, lumps, seed, device), lumps
