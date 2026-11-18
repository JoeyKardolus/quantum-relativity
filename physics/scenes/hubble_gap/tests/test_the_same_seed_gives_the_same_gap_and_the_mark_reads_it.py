"""The scene is a pure reading of a seeded run: the same seed gives the same numbers, another seed other numbers, and the mark judges them."""
from physics.scenes.hubble_gap import hubble_gap, passes
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_two_runs_of_one_seed_agree_to_the_last_digit():
    first = hubble_gap(8, steps=4, every=1, settings=HELD)
    again = hubble_gap(8, steps=4, every=1, settings=HELD)
    other = hubble_gap(8, seed=2, steps=4, every=1, settings=HELD)
    assert first["expansion"]["size"] == again["expansion"]["size"]
    assert first["early_rate"] == again["early_rate"] and first["late_rate"] == again["late_rate"]
    assert first["expansion"]["size"] != other["expansion"]["size"]
    growing = first["early_rate"] > 0 and first["late_rate"] > 0 and 1.04 <= first["late_over_early"] <= 1.14
    assert passes(first, True) == growing and not passes(first, False)
