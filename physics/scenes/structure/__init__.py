"""Structure scene: how the clustering of the energy volumes' energy changes over time.

  definitions/  rms_contrast, the pass mark
  logic/        structure (the scene)
"""
from physics.scenes.structure.definitions.pass_mark import LUMPS, REGION, passes
from physics.scenes.structure.definitions.rms_contrast import rms_contrast
from physics.scenes.structure.logic.structure import structure

__all__ = ["LUMPS", "REGION", "passes", "rms_contrast", "structure"]
