"""The scene times one size history three ways; a few steps of a held grid grow under 1%, which the mark calls a miss."""
import pytest

from physics.scenes.expansion import expansion
from physics.scenes.tests._gpu import HELD, needs_gpu
from physics.scenes.timescape import KNOWN_Q, passes, timescape


@needs_gpu
def test_the_size_history_is_the_expansion_scenes_and_no_observer_means_no_observer_clock():
    result = timescape(8, steps=6, every=2, observers=4, settings=HELD)
    grown = expansion(8, steps=6, every=2, settings=HELD)
    assert result["size"] == pytest.approx(grown["size"]), "one universe, read by two scenes"
    assert result["known_q"] == KNOWN_Q and result["observers"] == 4
    assert abs(result["size"][-1] / result["size"][0] - 1.0) < 0.01
    assert not passes(result, True)
    none = timescape(8, steps=2, every=1, observers=0, settings=HELD)
    assert none["observers"] == 0 and none["q_observer"] != none["q_observer"], "NaN: nobody to time it"
