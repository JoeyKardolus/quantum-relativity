"""Mouths: the 152 outside boxes of a block, each in one mouth, each with an inside box behind it."""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.axiom_4_crossing.definitions.mouth import mouth_box, mouth_size

wp.config.quiet = True


@wp.kernel
def _read_mouths(sizes: wp.array(dtype=wp.int32), boxes: wp.array2d(dtype=wp.vec3i)):
    code = wp.tid()
    sizes[code] = mouth_size(code)
    for slot in range(sizes[code]):
        boxes[code, slot] = mouth_box(code, slot)


def _mouths() -> tuple:
    sizes = wp.zeros(27, dtype=wp.int32, device="cpu")
    boxes = wp.zeros((27, 16), dtype=wp.vec3i, device="cpu")
    wp.launch(_read_mouths, dim=27, inputs=[sizes, boxes], device="cpu")
    side = np.array([-1, 0, 0, 0, 0, 1])                       # the step a box coordinate looks out along its axis
    looks = (side[:, None, None] + 1) * 9 + (side[None, :, None] + 1) * 3 + side[None, None, :] + 1
    return sizes.numpy(), boxes.numpy(), looks


def test_hand_worked_mouths():
    sizes, boxes, looks = _mouths()
    assert sizes[22] == 16 and sizes[25] == 4 and sizes[26] == 1 and sizes[13] == 0    # face, edge, corner, centre
    assert boxes[22, 0].tolist() == [5, 1, 1] and boxes[22, 5].tolist() == [5, 2, 2]   # the +x face
    assert boxes[22, 15].tolist() == [5, 4, 4] and boxes[4, 0].tolist() == [0, 1, 1]   # and the -x face
    assert boxes[25, 2].tolist() == [5, 5, 3] and boxes[26, 0].tolist() == [5, 5, 5]
    assert boxes[0, 0].tolist() == [0, 0, 0]
    assert looks[5, 2, 2] == 22 and looks[0, 0, 0] == 0 and looks[2, 3, 4] == 13 and looks[5, 5, 3] == 25


def test_every_outside_box_is_in_exactly_one_mouth_in_rising_order():
    sizes, boxes, looks = _mouths()
    assert sizes.sum() == 152 and (looks == 13).sum() == 64
    assert sorted(sizes.tolist()) == [0] + [1] * 8 + [4] * 12 + [16] * 6
    for code in range(27):
        if code == 13:
            continue
        mine = [tuple(box) for box in boxes[code, :sizes[code]].tolist()]
        assert mine == sorted(tuple(int(part) for part in box) for box in np.argwhere(looks == code))


def test_two_mouths_that_face_each_other_pair_box_for_box_on_the_same_free_coordinates():
    sizes, boxes = _mouths()[:2]
    for code in range(27):
        facing = 26 - code
        assert sizes[code] == sizes[facing]
        for slot in range(int(sizes[code])):
            here, there = boxes[code][slot], boxes[facing][slot]
            assert all(a == b or {int(a), int(b)} == {0, 5} for a, b in zip(here, there)), (code, slot)
