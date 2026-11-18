"""Size follows energy alone, and a joint's energy follows the balance: a harder push needs less energy to balance the same pull, so the joints start longer and the volumes the same."""
from dataclasses import replace

from physics.scenes.expansion import expansion, passes
from physics.engine import Settings
from physics.scenes.tests._gpu import needs_gpu

HELD = Settings(tear_length=200.0)


@needs_gpu
def test_sixteen_times_the_push_makes_the_joints_two_to_four_times_as_long_at_step_0():
    plain = expansion(8, steps=1, every=1, settings=HELD)
    hard = expansion(8, steps=1, every=1, settings=replace(HELD, push_strength=16.0 * HELD.push_strength))
    ratio = hard["joint_size"][0] / plain["joint_size"][0]
    print(f"joint size at step 0: {plain['joint_size'][0]:.0f}, with sixteen times the push {hard['joint_size'][0]:.0f}")
    assert 2.0 < ratio < 4.0, "measured 3.13: the push goes as the quanta to the fourth power, and a longer joint stands in more room"
    assert hard["volume_size"][0] == plain["volume_size"][0]
    assert not passes(plain, True, True), "one step is no expansion history"
