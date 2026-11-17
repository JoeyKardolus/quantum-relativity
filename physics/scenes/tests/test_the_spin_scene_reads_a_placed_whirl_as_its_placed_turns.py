"""The spin scene's known answer: one whole turn placed in a cell is read as that turn, with its sign, at every step.

Small and short (8 places, 300 steps, every rule on): a whirl of +1 and of -1 in an energy
volume and in a joint. The long run's numbers are the scorecard's, not this test's.
"""
from physics.scenes import spin
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_single_whirls_of_either_direction_read_their_placed_turns_for_300_steps():
    found = spin(8, steps=300, cases=(("volume", 1), ("volume", -1), ("joint", 1), ("joint", -1)))
    for case in found["cases"]:
        print(f"{'joint' if case['joint'] else 'volume'} {case['placed']:+d}: held {case['held_steps']} of 300 steps, "
              f"values seen {case['seen']}")
        assert case["seen"] == [case["placed"]] and case["held_steps"] == 300 and case["first_off"] == -1
    assert len({case["cell"] for case in found["cases"]}) == 4
