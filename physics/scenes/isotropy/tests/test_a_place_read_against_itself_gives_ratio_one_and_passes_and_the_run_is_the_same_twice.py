"""A pair whose two places are one place reads pull, law and flow ratios of exactly 1 and passes; the same seed reads the same twice."""
from physics.scenes.isotropy import isotropy, passes
from physics.scenes.tests._gpu import HELD, needs_gpu

SAME = (((1, 1, 0), (1, 1, 0)),)


@needs_gpu
def test_a_place_beside_itself_is_isotropic_by_construction():
    found = isotropy(16, steps=6, settle=3, pairs=SAME, diagonals=((1, 0, 0), (1, 1, 1)), settings=HELD)
    again = isotropy(16, steps=6, settle=3, pairs=SAME, diagonals=((1, 0, 0), (1, 1, 1)), settings=HELD)
    pair = found["pairs"][0]
    print(f"6 steps, a 4 times lump: pull depth at (1, 1, 0) {pair['pull_on']:.4e} (below 0: shallower than far away), "
          f"arrived there {pair['flow_on']:.0f} quanta, the lump kept {found['kept']:.4f} of its extra energy")
    assert pair["pull_on"] != 0.0 and pair["flow_on"] != 0.0, "six steps in, the lump's pull and flow have reached its neighbours"
    assert pair["pull_ratio"] == 1.0 and pair["flow_ratio"] == 1.0 and pair["law_pull_ratio"] == 1.0
    assert passes(found)
    assert found["diagonals"][0] == {"place": [1, 0, 0], "pull_ratio": 1.0, "flow_ratio": 1.0}
    assert found == again, "a seeded run reads the same twice"
