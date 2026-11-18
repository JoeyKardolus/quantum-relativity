"""Early and late are the means of the first and last quarter; the mark wants growth both times and a ratio of 1.04 to 1.14."""
import pytest

from physics.scenes.hubble_gap.definitions.early_late import early_late
from physics.scenes.hubble_gap.definitions.known import KNOWN_LATE_OVER_EARLY
from physics.scenes.hubble_gap.definitions.pass_mark import passes


def test_the_outer_quarters_of_eight_samples_are_two_samples_each():
    assert early_late([1.0, 3.0, 9.0, 9.0, 9.0, 9.0, 4.0, 8.0]) == (2.0, 6.0)
    assert early_late([5.0, 7.0]) == (5.0, 7.0) and early_late([5.0]) == (5.0, 5.0)
    with pytest.raises(ValueError):
        early_late([])


def test_the_known_gap_is_nine_percent_and_passes_its_own_mark():
    assert KNOWN_LATE_OVER_EARLY == pytest.approx(1.0905, abs=1e-4)
    assert passes({"early_rate": 67.4, "late_rate": 73.5, "late_over_early": KNOWN_LATE_OVER_EARLY}, True)


def test_shrinking_rates_a_ratio_off_the_band_and_a_fallen_universe_all_miss():
    assert not passes({"early_rate": -1.0, "late_rate": -1.09, "late_over_early": 1.09}, True)
    assert not passes({"early_rate": 1.0, "late_rate": 1.2, "late_over_early": 1.2}, True)
    assert not passes({"early_rate": 1.0, "late_rate": 1.0, "late_over_early": 1.0}, True)
    assert not passes({"early_rate": 1.0, "late_rate": 1.09, "late_over_early": 1.09}, False)
