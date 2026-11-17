"""A scene's `start` (a fit candidate's start amounts) reaches the universe it builds; without it the default start is used.

Joints twice as long at the start make the universe measurably bigger at step 0, in a
natural-start scene (expansion) and in a placed one (black hole reads the same start
through packed_start), and the unevenness a start names replaces the scene's own.
"""
from physics.scenes import expansion, structure
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_a_longer_joint_in_the_start_makes_the_scenes_universe_bigger_from_step_0():
    plain = expansion(8, steps=2, every=1)
    longer = expansion(8, steps=2, every=1, start={"joint_length": 2.0})
    print(f"size at step 0: {plain['size'][0]:.1f} with the default start, {longer['size'][0]:.1f} with joints 2 long")
    assert longer["joint_size"][0] > 1.9 * plain["joint_size"][0]
    assert longer["volume_size"][0] == plain["volume_size"][0]


@needs_gpu
def test_an_unevenness_named_in_the_start_replaces_the_scenes_own():
    plain = structure(8, steps=2, every=2)
    rough = structure(8, steps=2, every=2, start={"unevenness": 0.3})
    print(f"contrast at step 0: {plain['rms_contrast'][0]:.3f} default, {rough['rms_contrast'][0]:.3f} with unevenness 0.3")
    assert rough["rms_contrast"][0] > 2.0 * plain["rms_contrast"][0]
