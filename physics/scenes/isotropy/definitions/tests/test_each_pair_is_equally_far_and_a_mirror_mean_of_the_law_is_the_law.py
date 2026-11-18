"""Every pair's two places are equally far from the lump; the law is its own mirror mean; one marked place is found once in six."""
import numpy as np
import pytest

from physics.scenes.along_grid import box_field
from physics.scenes.isotropy.definitions.mirror_mean import mirror_mean
from physics.scenes.isotropy.definitions.pass_mark import passes
from physics.scenes.isotropy.definitions.places import DIAGONALS, PAIRS


def test_the_two_places_of_a_pair_are_at_the_same_straight_line_distance():
    assert [float(np.linalg.norm(on)) for on, _ in PAIRS] == [3.0, 5.0, 6.0, 7.0]
    for on, off in PAIRS:
        assert np.linalg.norm(on) == pytest.approx(np.linalg.norm(off))
        assert sum(index != 0 for index in on) == 1 and sum(index != 0 for index in off) >= 2
    assert DIAGONALS == ((3, 0, 0), (3, 3, 0), (3, 3, 3))


def test_a_field_with_the_cubes_symmetry_is_its_own_mirror_mean():
    law = box_field(8)
    assert mirror_mean(law, 0, (2, 2, 1)) == pytest.approx(law[2, 2, 1])
    assert mirror_mean(law, 0, (3, 0, 0)) == pytest.approx(law[3, 0, 0])
    shifted = np.roll(law, (2, 2, 2), axis=(0, 1, 2))
    assert mirror_mean(shifted, 2, (3, 0, 0)) == pytest.approx(law[3, 0, 0]), "counted from the centre, wrapping"


def test_one_marked_place_is_one_of_the_six_mirrors_of_an_axis_place():
    field = np.zeros((8, 8, 8))
    field[5, 4, 4] = 1.0
    assert mirror_mean(field, 4, (1, 0, 0)) == pytest.approx(1.0 / 6.0)
    assert mirror_mean(field, 4, (1, 1, 0)) == 0.0


def test_the_mark_reads_the_pull_beside_the_law_and_the_flow_beside_one():
    pair = {"pull_ratio": 0.93, "law_pull_ratio": 0.94, "flow_ratio": 1.05, "law_flow_ratio": 1.0}
    assert passes({"pairs": [pair]})
    assert not passes({"pairs": [pair, {**pair, "pull_ratio": 1.2}]}), "0.94 is isotropic here, 1.2 is not"
    assert not passes({"pairs": [{**pair, "flow_ratio": 0.84}]})
    assert not passes({"pairs": [{**pair, "pull_ratio": 0.0}]}) and not passes({"pairs": []})
