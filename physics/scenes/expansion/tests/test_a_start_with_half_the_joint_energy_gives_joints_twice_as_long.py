"""Size follows energy alone: a candidate's start with half the joints' quanta makes the joints' share of the size twice as big, the volumes' unchanged."""
from physics.scenes.expansion import expansion, passes
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_halved_joint_quanta_double_the_joint_size_at_step_0():
    plain = expansion(8, steps=1, every=1, settings=HELD)
    thin = expansion(8, steps=1, every=1, settings=HELD, start={"joint_quanta": 0.5 * HELD.joint_quanta})
    ratio = thin["joint_size"][0] / plain["joint_size"][0]
    print(f"joint size at step 0: {plain['joint_size'][0]:.0f} default, {thin['joint_size'][0]:.0f} with half the quanta")
    assert 1.95 < ratio < 2.05
    assert thin["volume_size"][0] == plain["volume_size"][0]
    assert thin["settings"]["joint_quanta"] == 50.0
    assert not passes(plain, True, True), "one step is no expansion history"
