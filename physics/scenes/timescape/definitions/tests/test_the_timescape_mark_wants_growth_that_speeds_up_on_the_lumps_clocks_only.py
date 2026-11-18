"""The mark passes growth that speeds up on the lumps' clocks and less so on the counter, and nothing else."""
from physics.scenes.timescape.definitions.known import KNOWN_Q
from physics.scenes.timescape.definitions.pass_mark import passes

GOOD = {"size": [100.0, 103.0], "q_observer": KNOWN_Q, "q_counter": -0.1}


def test_the_supernova_number_on_lump_clocks_passes():
    assert KNOWN_Q == -0.55 and passes(GOOD, True)


def test_no_growth_no_speed_up_no_gap_or_no_universe_is_a_miss():
    assert not passes({**GOOD, "size": [100.0, 100.5]}, True), "grew under 1%"
    assert not passes({**GOOD, "q_observer": 0.2}, True), "slows on the lumps' clocks"
    assert not passes({**GOOD, "q_counter": -0.52}, True), "the counter sees the same speed-up"
    assert not passes({**GOOD, "q_observer": float("nan")}, True)
    assert not passes(GOOD, False)
