"""A joint squeezed past the seal point passes no energy; the same joint relaxed does."""
from physics.abab import Settings, joint_rest_size, step
from physics.abab.tests._hand_built import hand_built


def test_a_joint_past_the_seal_point_passes_no_energy():
    """Two energy volumes, 1 and 3, through one joint at 0.55 of rest (depth 0.45, seal 0.4) and at 0.9."""
    settings = Settings.all_on()
    rest = float(joint_rest_size(0.05))
    for share, passes in ((0.55, False), (0.9, True)):
        state = hand_built([0, 1, 0], [1.0, 0.05, 3.0], [1.0, share * rest, 1.0], [(0, 1), (2, 1)])
        after = step(state, settings)
        assert after.collapses == 0
        moved = after.energy[0] != 1.0 or after.energy[2] != 3.0
        assert moved == passes
        assert after.energy[1] == 0.05
