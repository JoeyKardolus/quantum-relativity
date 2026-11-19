"""Of what a volume gives its neighbours, more crosses toward the side where the pull is deeper."""
import numpy as np

from physics.engine import calm_cells
from physics.engine.axiom_2_swirls.definitions.ring_boxes import cap_boxes
from physics.engine.axiom_3_structure.definitions.store import put
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.engine.axiom_4_crossing.logic.cross import cross

FRONT, BACK = cap_boxes(6, 1, 1)[0], cap_boxes(6, 1, 0)[0]  # a box of each cap along the line
LINE = [5 * steps for steps in range(5)]  # volume, joint, volume, joint, volume along direction 1


def _crossed(pulls: tuple, steps: int = 20):
    """Five cells of still fluid in a line, the middle volume fuller; the three volumes' pulls
    held as given. What the first and the last volume gained, and the store."""
    store = calm_cells(125, 5)
    store.side = 5
    store.cell_quanta.zero_()
    for cell, halves in zip(LINE, (2, -2, 4, -2, 2)):
        put(store.cell_quanta, cell, halves * 324_000)
        take_own(store, cell)
    quanta, phase = store.box_quanta.numpy(), store.box_phase.numpy()
    quanta[[0, 2, 4]], phase[0] = [[3000], [6000], [3000]], 0.0  # even and still, and in step:
    for joint in (1, 3):  # each volume on by the turn of the joint before it, cap to cap
        phase[joint + 1] = phase[joint - 1][0] + phase[joint][FRONT] - phase[joint][BACK]
    store.box_quanta.assign(quanta)
    store.box_phase.assign(np.angle(np.exp(1j * phase)).astype(np.float32))
    for cell, pull in zip(LINE[::2], pulls):
        put(store.cell_pull, cell, pull)
    for _ in range(steps):
        cross(store)
    held = store.cell_quanta.numpy()[LINE]
    assert held[[0, 2, 4]].sum() == 2_592_000 and (held[[1, 3]] == -648_000).all()  # exact books
    return held[0] - 648_000, held[4] - 648_000, store


def test_with_no_pull_or_an_even_pull_a_fuller_volume_gives_alike_to_both_sides():
    for pulls in ((0, 0, 0), (2, 2, 2)):
        before, after, _store = _crossed(pulls)
        assert before > 8_000 and abs(before - after) < 0.01 * before  # measured: 9,048 and 9,084


def test_more_crosses_toward_the_side_where_the_pull_is_deeper():
    before, after, _store = _crossed((1, 2, 3))  # deeper after the middle volume than before it
    assert after > 1.3 * before > 0 and after > 1.1 * _crossed((0, 0, 0))[1]  # measured: 5,544, 12,588
    mirrored = _crossed((3, 2, 1))
    assert mirrored[0] > 1.3 * mirrored[1]  # and the other way round alike (12,564, 5,574)
