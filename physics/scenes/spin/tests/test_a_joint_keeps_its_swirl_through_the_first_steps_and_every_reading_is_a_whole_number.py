"""On the held grid a joint still reads its whole turn in every layer for five steps; what a volume reads is reported, whole numbers either way."""
from physics.scenes.spin import passes, spin
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_five_steps_of_watching_two_volumes_and_two_joints():
    found = spin(8, steps=5, count=2, settings=HELD)
    for case in found["cases"]:
        print(f"{'joint' if case['joint'] else 'volume'} {case['cell']}: expected {case['expected']:+d}, held "
              f"{case['held_steps']} of 5 steps, first off at {case['first_off']}, values seen {case['seen']}")
        assert all(isinstance(turns, int) for turns in case["seen"]) and len(case["end"]) == 6
        assert (case["held_steps"] == 5) == (case["first_off"] == -1)
    joints = [case for case in found["cases"] if case["joint"]]
    assert all(case["held_steps"] == 5 for case in joints), "the engine's own finding: joints keep their swirl early on"
    assert passes(found) == all(case["held_steps"] == 5 for case in found["cases"])
