"""Sign law: only an energy volume makes and feels the pull, only a joint pushes, an empty id does nothing."""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_2_swirls.definitions.sign_law import feels_pull, makes_pull, pushes
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT, NO_CELL

wp.config.quiet = True


@wp.kernel
def _ask(kinds: wp.array(dtype=wp.int32), answers: wp.array2d(dtype=wp.int32)):
    index = wp.tid()
    if makes_pull(kinds[index]):
        answers[index, 0] = 1
    if feels_pull(kinds[index]):
        answers[index, 1] = 1
    if pushes(kinds[index]):
        answers[index, 2] = 1


def test_only_plus_makes_and_feels_the_pull_only_minus_pushes_and_an_empty_id_does_nothing():
    kinds = [ENERGY_VOLUME, JOINT, NO_CELL]
    answers = wp.zeros((3, 3), dtype=wp.int32, device="cpu")
    wp.launch(_ask, dim=3, inputs=[wp.array(kinds, dtype=wp.int32, device="cpu"), answers], device="cpu")
    assert answers.numpy().tolist() == [[1, 1, 0], [0, 0, 1], [0, 0, 0]]
