"""Scene: what the window shows, by how many cells hold their own boxes in it.

One cell is layer 1's energy volume. Three are a red volume, a blue joint and a red volume
in a line. Eight are four red volumes in a square with the four blue joints between them.
"""
import warp as wp

from physics.engine import one_cell, three_cells
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.lay_square import four_volumes
from physics.view.definitions.camera import DISTANCE, LINE_DISTANCE, SQUARE_DISTANCE
from physics.view.logic.line_streaks import line_streaks
from physics.view.logic.placed import placed
from physics.view.logic.streaks import streaks
from physics.view.logic.title_line import cells_title, title_line
from physics.view.logic.tracers import COUNT, Tracers

SCENES = {"one": 1, "crossing": 3, "grid": 8}  # a scene's name and its own cells
LAID = {1: (one_cell, DISTANCE), 3: (three_cells, LINE_DISTANCE),
        8: (four_volumes, SQUARE_DISTANCE)}


def lay_scene(cells: int, across: int, device: str) -> Store:
    """The scene's fresh start."""
    return LAID[cells][0](across, device=device)


def tracer_count(cells: int) -> int:
    """How many tracers the scene is drawn with: `COUNT` a cell, shared out past four cells."""
    return COUNT * min(cells, 4)


def drawn(store: Store, tracers: Tracers, cells: int) -> wp.array:
    """The tracers' tails as the card draws them now."""
    return streaks(tracers) if cells == 1 else line_streaks(tracers, store.across, placed(store))


def title(store: Store, cells: int) -> str:
    """The scene's numbers in words."""
    return title_line(store) if cells == 1 else cells_title(store)


def eye_distance(cells: int) -> float:
    """How far the eye starts from the middle of the scene."""
    return LAID[cells][1]
