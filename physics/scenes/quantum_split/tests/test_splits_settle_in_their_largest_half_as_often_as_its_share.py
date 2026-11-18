"""Over 512 splits the share that settles in the largest half's cell is the mean share those halves held, within 0.1."""
from physics.scenes.quantum_split import passes, quantum_split
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_a_cells_share_is_its_chance_over_512_splits():
    found = quantum_split(16, wait=3, settings=HELD)
    print(f"16 a side: {found['judged']} settlings judged; in the largest half {found['found_in_largest']:.3f}, "
          f"its mean share one step before {found['expected_in_largest']:.3f}")
    assert found["judged"] == 512 and found["whole_share"] == 1.0
    assert abs(found["found_in_largest"] - found["expected_in_largest"]) < 0.1
    assert found["found_in_largest"] < 1.0, "not always the largest half: the pointer jumps by share"
    assert passes(found) == (abs(found["found_in_largest"] - found["expected_in_largest"]) <= 0.05)
