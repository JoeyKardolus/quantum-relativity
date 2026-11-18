"""Colours as a whole: from a start's raw packings to what is drawn, with the engine's own proportions."""
import numpy as np

from physics.view.colours.definitions.palette import JOINT_BLUE, VOLUME_RED
from physics.view.colours.logic.cell_colours import cell_colours
from physics.view.colours.logic.relative_packing import relative_packing
from physics.view.colours.logic.start_marks import start_marks

KIND = np.array([1, 1, 1, 1, -1, -1, -1, -1])
START = np.array([700.0, 700.0, 700.0, 700.0, 8e-4, 8e-4, 8e-4, 8e-4])


def test_at_the_start_both_kinds_are_clear_alike_though_their_packings_differ_a_million_times():
    colours = cell_colours(KIND, relative_packing(KIND, START, start_marks(KIND, START)))
    assert colours[:4].tolist() == [[*VOLUME_RED, 0]] * 4 and colours[4:].tolist() == [[*JOINT_BLUE, 0]] * 4


def test_later_frames_are_read_against_the_start_so_a_lump_is_solid_and_a_thinned_cell_fades():
    marks = start_marks(KIND, START)
    later = np.array([2800.0, 350.0, 1750.0, 1e9, 8e-4, 3.2e-3, 2e-4, 2e-3])
    alpha = cell_colours(KIND, relative_packing(KIND, later, marks))[:, 3].tolist()
    assert alpha == [255, 0, 128, 255, 0, 255, 0, 128]
    again = cell_colours(KIND[:2], relative_packing(KIND[:2], later[:2], marks))[:, 3].tolist()
    assert again == alpha[:2], "the same packing, the same opacity, whatever else is in the frame"
