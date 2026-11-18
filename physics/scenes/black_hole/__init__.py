"""Black hole scene: how far round a packed core does nothing get away, and does that distance grow with the core's energy?

  definitions/  shell_ways (volumes sorted in shells round the core), held_shells (how
                far energy flows inward), depth_radius (how far the pull's depth
                reaches), the pass mark
  logic/        core_run (one placed run read), core_numbers (one run's numbers),
                black_hole (the scene)
"""
from physics.scenes.black_hole.definitions.depth_radius import depth_radius
from physics.scenes.black_hole.definitions.held_shells import held_shells
from physics.scenes.black_hole.definitions.pass_mark import ENERGIES, HIGHEST, LOWEST, SPAN, passes
from physics.scenes.black_hole.definitions.shell_ways import shell_ways
from physics.scenes.black_hole.logic.black_hole import black_hole
from physics.scenes.black_hole.logic.core_numbers import core_numbers
from physics.scenes.black_hole.logic.core_run import core_run

__all__ = ["ENERGIES", "HIGHEST", "LOWEST", "SPAN", "black_hole", "core_numbers", "core_run", "depth_radius",
           "held_shells", "passes", "shell_ways"]
