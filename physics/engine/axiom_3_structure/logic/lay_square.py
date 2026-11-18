"""Lay square: four energy volumes in a flat square with the four joints between neighbours.

The smallest piece of the grid that shows where joints meet: volumes at the corners of 3
by 3 places, a joint between each two neighbours, no cell in the middle. It is the edge of
a cut-out: the volumes' outer sides have no joints. Each volume is its neighbours' mirror.
"""
from physics.engine import LINE_PARTS, calm_cells
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS, QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import Store, put
from physics.engine.axiom_3_structure.logic.take_own import take_own

SQUARE = (3, 3, 1)
# Chosen before any run: the line's 3 to 1 on both diagonals, so every joint lies between a
# fuller and an emptier volume, and all four volumes are calm once even.
SQUARE_PARTS = (LINE_PARTS[0], LINE_PARTS[2], LINE_PARTS[2], LINE_PARTS[0])


def four_volumes(across: int = ACROSS, seed: int = 1, device: str = "cpu",
                 parts: tuple = SQUARE_PARTS, joint: int = LINE_PARTS[1]) -> Store:
    """Four volumes (`parts`, halves of a calm volume's quanta) and their four fresh joints."""
    store = calm_cells(9, 8, across, seed=seed, device=device, grid=SQUARE)
    for cell in range(9):
        down, along = divmod(cell, 3)
        corner = down % 2 == 0 and along % 2 == 0
        halves = parts[down + along // 2] if corner else joint * (down % 2 != along % 2)
        put(store.cell_quanta, cell, halves * QUANTA_A_BOX * across ** 3 // 2)
        put(store.cell_laid, cell, 4 + (down // 2 + along // 2) % 2)
        if halves:
            take_own(store, cell)
    return store
