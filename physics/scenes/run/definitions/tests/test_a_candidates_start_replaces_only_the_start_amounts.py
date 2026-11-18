"""A start replaces the named start amounts and nothing else; a name that is no start amount is refused."""
import pytest

from physics.engine import Settings
from physics.scenes.run.definitions.even_volume_quanta import even_volume_quanta
from physics.scenes.run.definitions.start_settings import start_settings


def test_a_named_amount_is_replaced_and_every_other_setting_is_kept():
    changed = start_settings(None, {"unevenness": 0.3, "joint_quanta": 50.0})
    assert changed.unevenness == 0.3 and changed.joint_quanta == 50.0
    assert changed.friction == Settings().friction and changed.volume_quanta == Settings().volume_quanta
    assert start_settings(None, None) == Settings()


def test_a_rule_setting_in_a_start_is_refused():
    with pytest.raises(ValueError):
        start_settings(None, {"friction": 0.9})


def test_an_even_volume_holds_its_boxes_times_the_mean_box():
    assert even_volume_quanta(Settings()) == 648000.0
    assert even_volume_quanta(start_settings(None, {"volume_quanta": 10.0})) == 2160.0
