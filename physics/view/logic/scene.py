"""Scene: what the window shows, by how many cells stand in its line.

One cell is layer 1's energy volume. Three are layer 2's crossing: a red volume, a blue
joint and a red volume, the left one holding three times the right one's energy.
"""
import warp as wp

from physics.engine import one_cell, three_cells
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.view.definitions.camera import DISTANCE, LINE_DISTANCE
from physics.view.logic.line_streaks import line_streaks, placed_line
from physics.view.logic.streaks import streaks
from physics.view.logic.title_line import line_title, title_line
from physics.view.logic.tracers import Tracers

SCENES = {"one": 1, "crossing": 3}  # a scene's name and the cells in its line


def lay_scene(cells: int, across: int, device: str) -> Store:
    """The scene's fresh start."""
    return one_cell(across, device=device) if cells == 1 else three_cells(across, device=device)


def drawn(store: Store, tracers: Tracers, cells: int) -> wp.array:
    """The tracers' tails as the card draws them now."""
    if cells == 1:
        return streaks(tracers)
    return line_streaks(tracers, store.across, placed_line(store, cells))


def title(store: Store, cells: int) -> str:
    """The scene's numbers in words."""
    return title_line(store) if cells == 1 else line_title(store)


def eye_distance(cells: int) -> float:
    """How far the eye starts from the middle of the scene."""
    return DISTANCE if cells == 1 else LINE_DISTANCE
