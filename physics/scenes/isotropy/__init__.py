"""Isotropy scene: are the pull and the flow round one lump the same along the grid's axes and off them?

  definitions/  PAIRS and DIAGONALS (the places read), mirror_mean, the pass mark
  logic/        isotropy (the scene)
"""
from physics.scenes.isotropy.definitions.mirror_mean import mirror_mean
from physics.scenes.isotropy.definitions.pass_mark import WITHIN, passes
from physics.scenes.isotropy.definitions.places import DIAGONALS, PAIRS
from physics.scenes.isotropy.logic.isotropy import isotropy

__all__ = ["DIAGONALS", "PAIRS", "WITHIN", "isotropy", "mirror_mean", "passes"]
