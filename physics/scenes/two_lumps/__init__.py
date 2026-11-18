"""Two lumps scene: does energy attract energy, and how does the pull between two lumps fall with distance?

  definitions/  newton_tilt (what one over distance squared gives for the same reading),
                extra_centre (where a lump's extra energy sits), the pass mark
  logic/        pair_run (one distance), two_lumps (the scene)
"""
from physics.scenes.two_lumps.definitions.extra_centre import extra_centre
from physics.scenes.two_lumps.definitions.newton_tilt import newton_tilt
from physics.scenes.two_lumps.definitions.pass_mark import FROM_VOLUMES, KEPT, POWER_WITHIN, falls, falls_off, passes
from physics.scenes.two_lumps.logic.pair_run import pair_run
from physics.scenes.two_lumps.logic.two_lumps import two_lumps

__all__ = ["FROM_VOLUMES", "KEPT", "POWER_WITHIN", "extra_centre", "falls", "falls_off", "newton_tilt", "pair_run",
           "passes", "two_lumps"]
