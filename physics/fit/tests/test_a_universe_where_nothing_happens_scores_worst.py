"""The fit's score is 0, the lowest it can be, for a universe where nothing happens, and above 0 only for change since the start."""
import pytest

from physics.fit.fit_score import fit_score

_STILL = {"room": 1000.0, "region_contrast": 0.5, "lump_share": 0.3, "sealed_share": 0.0, "lump_spread": 0.1,
          "lump_volumes": 40, "lumps": 5}


def _samples(rooms, **last):
    samples = [{**_STILL, "room": room, "tau": float(index)} for index, room in enumerate(rooms)]
    samples[-1].update(last)
    return samples


def test_a_universe_where_nothing_happens_scores_exactly_zero():
    scored = fit_score(_samples([1000.0] * 7))
    assert scored["score"] == 0.0 and scored["E"] == 0.0 and scored["S"] == 0.0


def test_losses_score_zero_too_never_below():
    scored = fit_score(_samples([1000.0, 990.0, 980.0, 970.0, 960.0, 950.0, 940.0], region_contrast=0.2,
                                lump_share=0.1, lump_spread=0.0))
    assert scored["score"] == 0.0


def test_growth_that_slows_then_speeds_up_beats_the_same_growth_that_only_slows():
    dipped = fit_score(_samples([1000.0, 1100.0, 1150.0, 1160.0, 1170.0, 1220.0, 1320.0]))
    slowing = fit_score(_samples([1000.0, 1150.0, 1220.0, 1270.0, 1300.0, 1315.0, 1320.0]))
    assert dipped["dip"] > 0.0 > slowing["dip"]
    assert 1.0 >= dipped["E"] > slowing["E"] > 0.0
    assert dipped["growth"] == pytest.approx(slowing["growth"])


def test_every_structure_part_counts_only_a_gain():
    for name in ("region_contrast", "lump_share", "sealed_share", "lump_spread"):
        scored = fit_score(_samples([1000.0] * 7, **{name: _STILL[name] + 0.2}))
        assert 0.0 < scored["S"] <= 0.25 and scored["E"] == 0.0
