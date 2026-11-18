"""The fit's score is 0 for a universe where nothing happens, about 0 for one that fell into a single cell or tore without end, has no cliff, and pays a runaway no more than steady growth."""
import math

import pytest

from physics.fit.score.logic.score_samples import score_samples

_STILL = {"size": 1000.0, "gap": 100.0, "alive": 1.0, "spent_share": 0.0, "region_contrast": 0.5, "lump_share": 0.3,
          "lump_spread": 0.1}
_GROWING = [1000.0, 1100.0, 1150.0, 1160.0, 1170.0, 1220.0, 1320.0]


def _samples(sizes, gaps=None, **last):
    samples = [{**_STILL, "size": size, "gap": _STILL["gap"] if gaps is None else gaps[index], "tau": float(index)}
               for index, size in enumerate(sizes)]
    samples[-1].update(last)
    return samples


def test_a_universe_where_nothing_happens_scores_exactly_zero():
    scored = score_samples(_samples([1000.0] * 7), held_share=_STILL["lump_share"])
    assert scored["score"] == 0.0 and scored["E"] == 0.0 and scored["S"] == 0.0 and scored["hold"] == 1.0


def test_losses_score_zero_too_never_below():
    scored = score_samples(_samples([1000.0, 990.0, 980.0, 970.0, 960.0, 950.0, 940.0], region_contrast=0.2,
                                    lump_share=0.1, lump_spread=0.0))
    assert scored["score"] == 0.0


def test_the_score_is_hold_times_the_mean_of_expansion_and_structure_with_known_parts():
    scored = score_samples(_samples(_GROWING, gaps=[value / 10.0 for value in _GROWING], lump_share=0.5, alive=0.5),
                           held_share=0.3)
    shape = math.tanh(3.2)
    assert scored["E_size"] == pytest.approx(shape) and scored["E_gap"] == pytest.approx(shape)
    assert scored["E"] == pytest.approx(shape) and scored["S"] == pytest.approx(math.tanh(0.5) / 4.0)
    assert scored["hold"] == 0.5
    assert scored["score"] == pytest.approx(0.5 * 0.5 * (shape + math.tanh(0.5) / 4.0))


def test_growth_that_slows_then_speeds_up_beats_the_same_growth_that_only_slows():
    dipped = score_samples(_samples(_GROWING))
    slowing = score_samples(_samples([1000.0, 1150.0, 1220.0, 1270.0, 1300.0, 1315.0, 1320.0]))
    assert dipped["dip"] > 0.0 > slowing["dip"]
    assert 0.5 >= dipped["E"] > slowing["E"] > 0.0, "the size alone is half of the expansion part"
    assert dipped["growth"] == pytest.approx(slowing["growth"])


def test_the_gap_between_lumps_is_the_other_half_of_the_expansion_part_and_none_without_two_lumps():
    both = score_samples(_samples(_GROWING, gaps=[value / 10.0 for value in _GROWING]))
    alone = score_samples(_samples(_GROWING, gaps=[None] * 7))
    assert both["E_gap"] == pytest.approx(both["E_size"]) and both["E"] == pytest.approx(2.0 * alone["E"])
    assert alone["E_gap"] == 0.0


def test_every_structure_part_is_a_quarter_of_the_structure_half():
    for name in ("region_contrast", "lump_share"):
        scored = score_samples(_samples([1000.0] * 7, **{name: _STILL[name] + 0.2}), held_share=_STILL["lump_share"])
        assert 0.0 < scored["S"] <= 0.25 and scored["E"] == 0.0
    held = score_samples(_samples([1000.0] * 7), held_share=_STILL["lump_share"] + 0.2)
    assert 0.0 < held["Sh"] and held["S"] == pytest.approx(held["Sh"] / 4.0)


def test_a_universe_that_fell_into_a_single_cell_scores_about_zero():
    whole = score_samples(_samples(_GROWING, lump_share=0.9, region_contrast=3.0), held_share=0.9)
    fallen = score_samples(_samples(_GROWING, lump_share=0.9, region_contrast=3.0, alive=1.0 / 1728.0), held_share=0.9)
    assert whole["score"] > 0.3 and fallen["score"] < 0.001


def test_a_universe_that_tears_without_end_scores_about_zero_whatever_its_size_did():
    torn = score_samples(_samples([1000.0, 1100.0, 1300.0, 2000.0, 9000.0, 1e6, 5e7], spent_share=0.97))
    assert torn["E_size"] > 0.4 and torn["score"] < 0.02


def test_there_is_no_cliff_the_score_falls_in_a_straight_line_as_volumes_fall_in_or_are_spent():
    full = score_samples(_samples(_GROWING))["score"]
    for name, values in (("alive", (1.0, 0.75, 0.5, 0.25, 0.0)), ("spent_share", (0.0, 0.25, 0.5, 0.75, 1.0))):
        scores = [score_samples(_samples(_GROWING, **{name: value}))["score"] for value in values]
        assert scores == pytest.approx([full * left for left in (1.0, 0.75, 0.5, 0.25, 0.0)])


def test_a_runaway_earns_no_more_than_steady_growth():
    steady = score_samples(_samples([1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0]))
    runaway = score_samples(_samples([1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1e5, 1e9]))
    assert runaway["score"] <= steady["score"] * 1.01


def test_too_few_samples_and_a_clock_that_does_not_advance_are_refused():
    with pytest.raises(ValueError, match="at least 4 samples"):
        score_samples(_samples([1000.0, 1100.0, 1200.0]))
    stuck = _samples(_GROWING)
    stuck[3]["tau"] = stuck[2]["tau"]
    with pytest.raises(ValueError, match="must advance"):
        score_samples(stuck)
