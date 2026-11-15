"""A joint collapse and a volume split keep ABAB and the total energy."""
import numpy as np

from physics.abab import KIND_A, KIND_B, Settings, collapse_joints, new_state, split_volumes, step
from physics.abab.tests._hand_built import hand_built


def test_a_collapse_and_a_split_keep_abab_and_total_energy():
    """By hand: a collapse merges exactly and keeps the other joints; a split shares exactly; and in a run."""
    # energy volumes 0, 1, 2; joint 3 between 0 and 1, joint 4 between 1 and 2, joint 5 between 0 and 2
    state = hand_built([0, 0, 0, 1, 1, 1], [1.0, 2.0, 0.5, 0.05, 0.04, 0.06], [1.0, 0.5, 2.0, 0.03, 0.03, 0.03],
                       [(0, 3), (1, 3), (1, 4), (2, 4), (0, 5), (2, 5)])
    collapsed = collapse_joints(state, np.array([False, False, False, True, False, False]))
    assert collapsed.kind.tolist() == [KIND_A, KIND_A, KIND_B, KIND_B]
    assert collapsed.contacts.tolist() == [[0, 2], [0, 3], [1, 2], [1, 3]]
    assert collapsed.energy[0] == 1.0 + 2.0 + 0.05 and collapsed.collapses == 1
    assert np.isclose(collapsed.size[0], 1.5 * (3.0 / 3.05) ** 3)
    assert collapsed.total_energy() == state.total_energy()

    split = split_volumes(collapsed, trigger=1.2)
    assert split.splits == 2 and len(split.kind) == 8
    kinds = split.kind[split.contacts]
    assert (kinds[:, 0] == KIND_A).all() and (kinds[:, 1] == KIND_B).all()
    assert np.isclose(split.total_energy(), state.total_energy(), rtol=1e-15, atol=0)
    assert np.isclose(split.size.sum(), collapsed.size.sum(), rtol=1e-15, atol=0)
    assert all((split.contacts[:, 1] == joint).sum() == 2 for joint in (5, 7))

    run, _ = new_state(300, 4, 0.5)
    start = run.total_energy()
    settings = Settings.all_on(push_limit=0.85, split_size=1.5)
    for _ in range(40):
        run = step(run, settings)
        kinds = run.kind[run.contacts]
        assert (kinds[:, 0] == KIND_A).all() and (kinds[:, 1] == KIND_B).all()
        assert abs(run.total_energy() - start) <= 1e-12 * start
    assert run.collapses > 0 and run.splits > 0
