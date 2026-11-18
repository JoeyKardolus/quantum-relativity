"""The natural start holds no lump (none of 64 volumes is 2 even volumes' worth at unevenness 0.1), misses nothing, and is no structure."""
from physics.scenes.structure import passes, structure
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_no_lump_at_the_start_and_the_field_misses_nothing_while_the_grid_stands():
    result = structure(8, steps=3, every=1, settings=HELD)
    assert result["lumps"][0] == 0 and result["largest_region"][0] == 0
    assert result["left_out_share"] == [0.0, 0.0, 0.0, 0.0]
    assert len(result["power_growth"]) == 2 and all(growth > 0.0 for growth in result["power_growth"])
    assert not passes(result, True, True)
