"""contact_pairs finds which cells touch across a box face and how many faces they share; edges and corners do not touch."""
import numpy as np

from physics.measure import contact_pairs


def test_two_blocks_side_by_side_share_one_full_face_and_a_corner_touch_counts_nothing():
    owner = np.full((6, 4, 4), -1)
    owner[0:3] = 0
    owner[3:6] = 1
    low, high, faces = contact_pairs(owner)
    assert low.tolist() == [0] and high.tolist() == [1] and faces.tolist() == [16]
    corner = np.full((2, 2, 2), -1)
    corner[0, 0, 0], corner[1, 1, 1] = 3, 5
    assert contact_pairs(corner)[0].size == 0
