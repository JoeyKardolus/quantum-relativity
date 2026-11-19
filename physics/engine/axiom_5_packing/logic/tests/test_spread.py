"""The pull is handed on from cell to cell: deepest at a fuller volume, fading away from it, a cell a step."""
import numpy as np
import pytest

from physics.engine import three_cells
from physics.engine.axiom_5_packing.logic.lumps import fuller, grown, region
from physics.engine.axiom_5_packing.logic.spread import spread


def _handed(store, steps: int) -> list:
    """The pull at every place after each of `steps` spreadings from none; nothing else moves."""
    store.cell_pull.zero_()
    seen = []
    for _ in range(steps):
        spread(store)
        seen.append(store.cell_pull.numpy().copy())
    return seen


@pytest.fixture(scope="module")
def felt():
    """A grown patch held still: the pull as it stands, then with the middle volume made fuller."""
    store = grown(3)
    row = [region(store, (0, out, 0))[0] for out in range(3)]  # the middle volume and two on
    plain = _handed(store, 150)
    fuller(store, row[:1])
    return np.array(plain)[:, row], np.array(_handed(store, 150))[:, row], store


def test_the_pull_is_deeper_at_a_fuller_volume_and_fades_away_from_it(felt):
    plain, more, _store = felt
    gained = more[-1] - plain[-1]
    assert gained[0] > gained[1] > gained[2] > 0   # deepest where the energy is, less a volume on
    assert more[-1][0] > more[-1][1] > more[-1][2]
    assert gained[0] - gained[1] > 2 * (gained[1] - gained[2])  # and its slope falls too


def test_the_pull_reaches_the_nth_neighbour_no_sooner_than_n_steps(felt):
    plain, more, _store = felt
    first = [int(np.flatnonzero(more[:, out] != plain[:, out])[0]) + 1 for out in range(3)]
    assert first == [1, 3, 5]  # a joint lies between two volumes: two cells a volume


def test_a_volume_s_pull_is_its_own_energy_and_a_sixth_of_each_side_s_joint_s():
    pair = three_cells(parts=(2, -2, 2))
    seen = np.array(_handed(pair, 40))[:, [0, 3, 6]]
    assert seen[0].tolist() == [1, 0, 1] and seen[1].tolist() == [1, 1, 1]  # the joint: their mean
    assert np.allclose(seen[2], [1 + 1 / 6, 1, 1 + 1 / 6]) and np.allclose(seen[-1], 1.2)
    apart = three_cells(parts=(3, 0, 1))  # no joint between: nothing is handed on
    assert np.allclose(np.array(_handed(apart, 10))[-1][[0, 3, 6]], [1.5, 0, 0.5])
