"""The four structure parts are gains since the start: known tanh values, 0 for a loss, and the spread counts only as far as energy gathered."""
import math

import pytest

from physics.fit.score.definitions import score_marks
from physics.fit.score.definitions.structure_parts import structure_parts

_START = {"region_contrast": 0.5, "lump_share": 0.3, "lump_spread": 0.1}


def test_the_scales_are_the_ones_written_down():
    marks = (score_marks.GROWTH_SCALE, score_marks.REGION_SCALE, score_marks.LUMP_SCALE, score_marks.HELD_SCALE,
             score_marks.SPREAD_SCALE, score_marks.MIN_SAMPLES)
    assert marks == (0.1, 1.0, 0.4, 0.4, 0.3, 4)


def test_nothing_changed_gives_four_zeros_and_there_is_no_part_for_sealing():
    parts = structure_parts(_START, _START, held_share=0.3)
    assert parts == {"Sr": 0.0, "Sl": 0.0, "Sh": 0.0, "Sp": 0.0}


def test_each_gain_is_tanh_of_its_rise_over_its_scale():
    last = {"region_contrast": 1.0, "lump_share": 0.5, "lump_spread": 0.9}
    parts = structure_parts(_START, last, held_share=0.4)
    assert parts["Sr"] == pytest.approx(math.tanh(0.5 / 1.0)) and parts["Sl"] == pytest.approx(math.tanh(0.2 / 0.4))
    assert parts["Sh"] == pytest.approx(math.tanh(0.1 / 0.4))
    assert parts["Sp"] == pytest.approx(math.tanh(0.8 / 0.3) * math.tanh(0.2 / 0.4))


def test_losses_count_nothing_and_a_wider_spread_counts_nothing_when_the_lumps_only_melted():
    last = {"region_contrast": 0.2, "lump_share": 0.05, "lump_spread": 0.9}
    assert structure_parts(_START, last, held_share=0.05) == {"Sr": 0.0, "Sl": 0.0, "Sh": 0.0, "Sp": 0.0}
