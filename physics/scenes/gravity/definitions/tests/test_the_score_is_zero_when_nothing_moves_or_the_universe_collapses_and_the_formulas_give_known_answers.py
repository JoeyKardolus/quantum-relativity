"""The phase 1 score, the lump's centre, the orbit's kick and the scenes' lumps on plain numbers."""
import math

import numpy as np
import pytest

from physics.scenes.gravity.definitions.lump import Lump
from physics.scenes.gravity.definitions.lump_centre import lump_centre
from physics.scenes.gravity.definitions.orbit_kick import SWIRL_SPEED, fall_rate, orbit_kick
from physics.scenes.gravity.definitions.pass_mark import gravity_score, passes
from physics.scenes.gravity.definitions.scene_lumps import scene_lumps

GOOD = {"stand_steps": 100, "stand_collapses": 0, "stand_tears": 0, "fall_closed": 0.3, "fall_kept": 0.9,
        "fall_sped_up": 1.0, "lone_drift": 0.03, "coast_travelled": 5.0, "coast_early_speed": 0.01,
        "coast_late_speed": 0.009, "coast_kept": 0.8, "pull_power": -2.0, "law_power": -2.1, "orbit_turns": 1.2,
        "orbit_kept": 0.7, "orbit_big_kept": 0.9}


def test_a_universe_in_which_gravity_works_scores_near_one_and_passes():
    parts = gravity_score(GOOD)
    assert parts["stands"] == 1.0 and parts["falls"] == pytest.approx(0.9 * 0.9) and parts["orbits"] == 1.0
    assert parts["coasts"] == pytest.approx(0.8 * 0.9) and parts["fades"] == pytest.approx(0.8)
    assert parts["score"] == pytest.approx((0.81 + 0.72 + 0.8 + 1.0) / 4.0) and passes(GOOD)


def test_nothing_moving_scores_zero_and_so_does_a_collapse_however_well_the_rest_goes():
    still = dict(GOOD, fall_closed=0.0, coast_travelled=0.0, orbit_turns=0.0)
    assert gravity_score(still)["score"] == 0.0 and gravity_score(still)["fades"] > 0.0
    assert gravity_score(dict(GOOD, stand_collapses=1))["score"] == 0.0
    assert gravity_score(dict(GOOD, stand_tears=3))["score"] == 0.0 and gravity_score({})["score"] == 0.0


def test_a_lump_that_keeps_under_half_drifts_by_itself_or_slows_down_is_no_body():
    assert gravity_score(dict(GOOD, fall_kept=0.4))["falls"] == 0.0
    assert gravity_score(dict(GOOD, lone_drift=0.3))["falls"] == 0.0, "a drift as large as the closing is no fall"
    assert gravity_score(dict(GOOD, fall_sped_up=0.0))["falls"] == pytest.approx(0.5 * 0.81)
    assert gravity_score(dict(GOOD, coast_late_speed=0.0))["coasts"] == 0.0
    assert gravity_score(dict(GOOD, orbit_kept=0.2))["orbits"] == 0.0 and not passes(dict(GOOD, orbit_turns=0.3))
    assert gravity_score(dict(GOOD, pull_power=float("nan")))["fades"] == 0.0


def test_a_lump_s_centre_is_the_mean_place_of_its_extra_the_short_way_round():
    energy = np.full((8, 8, 8), 10.0)
    energy[7, 2, 2], energy[0, 2, 2] = 30.0, 30.0            # one lump lying across the edge
    energy[4, 4, 4] = 50.0
    centre, amount = lump_centre(energy, (7.0, 2.0, 2.0), 1.5)
    assert centre == pytest.approx((7.5, 2.0, 2.0)) and amount == 40.0
    assert lump_centre(np.full((4, 4, 4), 3.0), (1.0, 1.0, 1.0), 1.5) == ((1.0, 1.0, 1.0), 0.0)


def test_the_kick_is_the_speed_a_circle_asks_for_and_the_scenes_place_what_they_say():
    assert fall_rate(-0.004, 0.25) == pytest.approx(2.0 * 0.0625 * 0.004)
    speed = math.sqrt(fall_rate(0.004, 0.25) * 18.0)
    assert orbit_kick(0.004, 18.0, 0.25) == pytest.approx(math.asin(speed / (SWIRL_SPEED * 0.5)))
    assert orbit_kick(1e9, 18.0, 0.25) == pytest.approx(math.pi / 2.0)
    first, second = scene_lumps("two_lumps", 24)
    assert second.centre[0] - first.centre[0] == 24.0 and first.kick == (0.0, 0.0, 0.0)
    assert scene_lumps("coast", 24)[0].kick[0] > 0.0 and len(scene_lumps("orbit", 32)) == 2
    with pytest.raises(ValueError):
        scene_lumps("two_lumps", 16)
    with pytest.raises(ValueError):
        scene_lumps("moon", 24)
    with pytest.raises(ValueError):
        Lump((0.0, 0.0, 0.0), 0.0, 1.0)
