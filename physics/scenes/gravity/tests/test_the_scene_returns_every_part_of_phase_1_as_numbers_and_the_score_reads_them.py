"""The whole gravity scene on a short real run: every part is there, and the score reads it."""
import math

from physics.engine import Settings
from physics.scenes.gravity import gravity, gravity_score
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_a_short_run_returns_the_five_parts_beside_the_law_and_scores_between_0_and_1():
    numbers = gravity(24, steps=4, every=2, settings=Settings(friction=0.01, clock_packing=1e12, tear_length=1e6))
    for name in ("stand_steps", "pull_power", "law_power", "lone_drift", "fall_closed", "fall_kept",
                 "coast_travelled", "coast_early_speed", "orbit_turns", "orbit_kick", "joint_length", "volume_length"):
        assert math.isfinite(numbers[name]), name
    assert numbers["stand_steps"] == 4 and numbers["stand_collapses"] == 0 and numbers["start_volumes"] == 1728
    assert len(numbers["fall_apart"]) == 3 and abs(numbers["fall_apart"][0] - 4.0) < 0.05
    assert len(numbers["pull_tilts"]) == len(numbers["law_tilts"]) == 4 and all(tilt > 0.0 for tilt in numbers["pull_tilts"])
    parts = gravity_score(numbers)
    assert all(0.0 <= value <= 1.0 for value in parts.values())
