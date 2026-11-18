"""Size law: a cell's length is a constant over its quanta, for volumes and joints alike.

More energy, smaller. The one constant is the quanta of a calm volume, so a calm volume
is 1 long; a cell with no quanta is empty space and has no length to give.
"""
from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX


def size_of(quanta: int, across: int) -> float:
    """The length of a cell that holds `quanta` (either sign), in calm volumes."""
    return QUANTA_A_BOX * across ** 3 / abs(quanta) if quanta else float("inf")
