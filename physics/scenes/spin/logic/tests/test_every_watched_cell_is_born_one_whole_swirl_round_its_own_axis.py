"""At the start every watched cell, volume and joint, reads one whole turn of its own sense in all six layers across its swirl axis."""
from physics.scenes.spin.logic.spin import spin
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_two_volumes_and_two_joints_are_born_whole_swirls():
    found = spin(8, steps=0, count=2, settings=HELD)
    assert [case["cell"] for case in found["cases"]] == [0, 292, 1, 293]
    assert [case["joint"] for case in found["cases"]] == [0, 0, 1, 1]
    for case in found["cases"]:
        assert case["expected"] in (1, -1) and case["axis"] in (0, 1, 2)
        assert case["born"] == [case["expected"]] * 6, "one whole turn, the same in every layer"
        assert case["held_steps"] == 0 and case["first_off"] == -1 and case["end"] == case["born"]
