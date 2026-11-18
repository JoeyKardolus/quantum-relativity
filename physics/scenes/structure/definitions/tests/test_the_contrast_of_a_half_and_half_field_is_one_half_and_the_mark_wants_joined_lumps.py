"""A field of ones and threes has contrast one half; the mark wants more contrast, 5 lumps, a region of 8 and a standing universe."""
import numpy as np
import pytest

from physics.scenes.structure.definitions.pass_mark import passes
from physics.scenes.structure.definitions.rms_contrast import rms_contrast

GOOD = {"rms_contrast": [0.1, 0.4], "lumps": [0, 12], "largest_region": [0, 9]}


def test_contrast_is_the_rms_of_the_field_over_its_mean_less_one():
    assert rms_contrast(np.array([1.0, 3.0, 1.0, 3.0])) == pytest.approx(0.5)
    assert rms_contrast(np.full((4, 4, 4), 7.0)) == 0.0
    with pytest.raises(ValueError):
        rms_contrast(np.zeros(8))


def test_the_mark_passes_lumps_joined_over_a_region_and_misses_anything_less():
    assert passes(GOOD, True, True)
    assert not passes({**GOOD, "rms_contrast": [0.1, 0.05]}, True, True), "it evened out"
    assert not passes({**GOOD, "lumps": [0, 4]}, True, True)
    assert not passes({**GOOD, "largest_region": [0, 7]}, True, True), "lumps, but single cells"
    assert not passes(GOOD, True, False) and not passes(GOOD, False, True)
