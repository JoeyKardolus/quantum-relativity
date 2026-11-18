"""Inside pull: the eight corners a box stands between and how much each counts, against hand-worked values."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_5_packing.definitions.inside_pull import corner_code, corner_weight

wp.config.quiet = True


@wp.kernel
def _corners(codes: wp.array4d(dtype=wp.int32), weights: wp.array4d(dtype=wp.float64)):
    x, y, z, corner = wp.tid()
    codes[x, y, z, corner] = corner_code(x, y, z, corner)
    weights[x, y, z, corner] = corner_weight(x, y, z, corner)


@pytest.fixture(scope="module")
def corners() -> tuple:
    codes = wp.zeros((6, 6, 6, 8), dtype=wp.int32, device="cpu")
    weights = wp.zeros((6, 6, 6, 8), dtype=wp.float64, device="cpu")
    wp.launch(_corners, dim=(6, 6, 6, 8), outputs=[codes, weights], device="cpu")
    return codes.numpy(), weights.numpy()


def test_the_eight_weights_of_every_box_sum_to_one_and_none_is_below_zero(corners):
    codes, weights = corners
    assert np.allclose(weights.sum(axis=3), 1.0, atol=1e-15) and (weights >= 0.0).all()


def test_corner_0_is_the_middle_of_the_cell_and_the_others_lie_toward_the_side_the_box_is_nearer_to(corners):
    codes, weights = corners
    assert (codes[..., 0] == 13).all()
    assert codes[5, 2, 2, 1] == 22 and codes[0, 2, 2, 1] == 4            # bit 0: along x, toward +x or -x
    assert codes[5, 2, 2, 2] == 10 and codes[5, 3, 2, 2] == 16           # bit 1: along y
    assert codes[5, 2, 2, 7] == 2 * 9 + 0 * 3 + 0 and codes[0, 5, 5, 7] == 0 * 9 + 2 * 3 + 2


def test_a_box_at_a_side_counts_only_that_side_and_a_box_near_the_middle_counts_mostly_the_middle(corners):
    codes, weights = corners
    assert weights[5, 2, 2].tolist() == pytest.approx([0.0, 0.64, 0.0, 0.16, 0.0, 0.16, 0.0, 0.04])
    assert weights[3, 2, 2, 0] == pytest.approx(0.8 ** 3) and weights[3, 2, 2, 1] == pytest.approx(0.2 * 0.64)
    assert weights[5, 5, 5].tolist() == pytest.approx([0.0] * 7 + [1.0])


def test_the_pull_a_box_feels_rises_evenly_from_the_middle_to_a_side(corners):
    codes, weights = corners
    toward_x = [weights[x, 2, 2, [1, 3, 5, 7]].sum() for x in (3, 4, 5)]
    assert toward_x == pytest.approx([0.2, 0.6, 1.0])
