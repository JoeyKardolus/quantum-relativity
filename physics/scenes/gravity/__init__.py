"""Gravity scene: phase 1 of the fit. Does the universe stand, do lumps fall, coast, pull by the law and go round?

  definitions/  lump (a placed lump), scene_lumps (what each scene places), lump_centre
                (where a lump's extra energy sits), orbit_kick (the sideways kick a
                circle asks for), the score and pass mark
  logic/        lumps_start and scene_start (the starts), volume_energy, lump_path
                (lumps followed as bodies), gravity (the scene)
"""
from physics.scenes.gravity.definitions.lump import Lump
from physics.scenes.gravity.definitions.lump_centre import lump_centre
from physics.scenes.gravity.definitions.orbit_kick import fall_rate, orbit_kick
from physics.scenes.gravity.definitions.pass_mark import gravity_score, passes
from physics.scenes.gravity.definitions.scene_lumps import SCENES, scene_lumps
from physics.scenes.gravity.logic.gravity import gravity
from physics.scenes.gravity.logic.lump_path import lump_path
from physics.scenes.gravity.logic.lumps_start import lumps_start
from physics.scenes.gravity.logic.scene_start import scene_start
from physics.scenes.gravity.logic.volume_energy import volume_energy

__all__ = ["Lump", "SCENES", "fall_rate", "gravity", "gravity_score", "lump_centre", "lump_path", "lumps_start",
           "orbit_kick", "passes", "scene_lumps", "scene_start", "volume_energy"]
