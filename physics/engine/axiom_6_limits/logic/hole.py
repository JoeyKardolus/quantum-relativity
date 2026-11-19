"""Hole: a small grown patch whose middle is packed until it gives way, to watch a collapse.

In a grown universe nothing is as deep as the first volume, and the first volume is under
the limit, so no joint collapses there by itself. This start stages one: a patch grown by
its leak, then the first volume and the six at its sides each packed to hold `times` what
a universe's first volume holds (`FIRST_QUANTA`: no new number). Each one's pull reaches
the others a step late; three steps on all seven are past the limit, and so are the volumes
a thin joint off them: 54 joints flip to plus at once and no more, and nothing leaves the
merged volume. Only a scene's start stages; no step does.
"""
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.first_volume import FIRST_QUANTA
from physics.engine.axiom_5_packing.logic.lumps import grown, region
from physics.engine.axiom_5_packing.logic.staging import packed


def hole(across: int = ACROSS, device: str = "cpu", layers: int = 3, times: float = 1.0) -> Store:
    """A patch grown `layers` layers out, its middle region packed to `times` a first volume's quanta each."""
    store = grown(layers, layers + 2, across, device)
    packed(store, region(store, (0, 0, 0)), int(times * FIRST_QUANTA))
    return store
