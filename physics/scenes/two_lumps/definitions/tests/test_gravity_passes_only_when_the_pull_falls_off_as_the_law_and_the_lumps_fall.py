"""The gravity mark has two parts, the fall-off and the falling, and passes only with both."""
from physics.scenes.two_lumps.definitions.pass_mark import falls, falls_off, passes

DISTANCE = [600.0, 900.0, 1200.0, 1800.0]
GOOD = {"apart": [2, 3, 4, 6], "distance": DISTANCE, "newton": [1.0, 0.44, 0.25, 0.11],
        "pull": [8.0, 2.0 * 0.44, 2.0 * 0.25, 2.0 * 0.11], "closing": [0.3, 0.2, 0.2, 0.1],
        "closing_alone": [0.01, -0.02, 0.0, 0.01], "kept": [0.9, 0.8, 0.8, 0.7]}


def test_a_pull_that_follows_the_law_from_three_volumes_on_and_lumps_that_close_in_pass():
    assert falls_off(GOOD), "the nearest distance, 2 volumes, is not judged"
    assert falls(GOOD) and passes(GOOD)


def test_each_part_can_miss_alone():
    flat = {**GOOD, "pull": [1.0, 1.0, 0.9, 0.8]}
    assert not falls_off(flat) and falls(flat) and not passes(flat)
    pushed = {**GOOD, "pull": [1.0, 0.4, -0.1, 0.1]}
    assert not falls_off(pushed), "a pull below 0 has no power"
    spread = {**GOOD, "kept": [0.9, 0.8, 0.4, 0.7]}
    assert falls_off(spread) and not falls(spread), "a lump that spread away has not fallen"
    still = {**GOOD, "closing": [0.3, 0.2, 0.015, 0.1]}
    assert not falls(still), "closing in by less than a lone lump's scatter"
