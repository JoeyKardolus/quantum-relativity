"""Before any step a placed lump holds all its extra energy, no energy has arrived anywhere else, no pull has spread, and the law beside is the cube's."""
import pytest

from physics.scenes.along_grid import box_field
from physics.scenes.isotropy.logic.isotropy import isotropy
from physics.scenes.tests._gpu import HELD, needs_gpu

PAIR = (((3, 0, 0), (2, 2, 1)),)


@needs_gpu
def test_the_start_of_an_isotropy_run_is_a_lump_and_nothing_else():
    found = isotropy(16, steps=0, settle=0, pairs=PAIR, diagonals=((2, 0, 0), (2, 2, 0)), settings=HELD)
    pair = found["pairs"][0]
    law = box_field(8)
    assert found["kept"] == pytest.approx(1.0, abs=0.02)
    assert pair["flow_on"] == 0.0 and pair["flow_off"] == 0.0, "the twin has the same jitter: nothing but the lump differs"
    assert pair["pull_on"] == 0.0 and pair["pull_off"] == 0.0, "the pull has not heard of the lump"
    assert pair["law_pull_ratio"] == pytest.approx(law[2, 2, 1] / law[3, 0, 0]) and pair["law_pull_ratio"] < 0.97
    assert pair["distance"] == 3.0 and pair["law_flow_ratio"] == 1.0
    with pytest.raises(ValueError):
        isotropy(16, steps=0, pairs=(((4, 0, 0), (4, 0, 0)),), settings=HELD)
