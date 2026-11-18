"""The lump is one energy volume of the start lattice followed through every sample; after a few steps it is not yet 2 even volumes' worth."""
from physics.scenes.bound_lump import bound_lump, passes
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_the_lump_is_a_start_energy_volume_and_the_mark_refuses_a_volume_that_is_no_lump():
    result = bound_lump(8, steps=4, every=1, settings=HELD)
    place = (result["lump"] // 64, (result["lump"] // 8) % 8, result["lump"] % 8)
    assert all(index % 2 == 0 for index in place), "an energy volume sits where all three indices are even"
    assert all(length > 0.0 for length in result["lump_length"]) and len(result["time"]) == 5
    assert result["is_lump"] == int(result["lump_quanta_factor"] >= 2.0)
    assert 0.9 < result["size_growth"] < 1.1
    assert not passes(result, True, True) or result["is_lump"] == 1
