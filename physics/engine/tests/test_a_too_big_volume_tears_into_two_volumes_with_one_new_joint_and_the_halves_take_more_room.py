"""A volume grown past split_across tears in two; the tear is one new joint; the halves then take more room.

Bible rule 8 and rule 4 (owner, 2026-11-16: a volume split makes new room). A lattice of
3 cells a side holds one energy volume of 10**3 boxes wrapped by 26 joints. With
split_across 9 boxes it is too big at the first step: two lattice planes go in along
one axis, so the lattice has 5 cells on that axis and two volumes with the new joint,
the cell halfway between them, as their one shared joint. In the boxes: every box's
sign matches its cell's kind, no box of one half has a box of the other among its 26
neighbours, the new joint holds minus boxes, every volume stays wrapped and the joints
stay one web. The amount ignoring sign is exact, and what the tear turned minus is the
minus gained. Then, splits off (split_across 0, so the two halves are watched alone),
after 75 more steps the halves hold more boxes than the parent did: each holds about
half the quanta, so it wants about eight times the room.
"""
import numpy as np

from physics.engine import Settings, cells, host_boxes, lattice_kinds, new_universe, stats, step, totals
from physics.engine.tests._gpu import needs_gpu


def _touching(owner: np.ndarray, quanta: np.ndarray, first: int, second: int) -> int:
    """Box pairs, one in each of two cells, that are 26-neighbours."""
    one = np.pad((owner == first) & (quanta > 0), 1)
    two = (owner == second) & (quanta > 0)
    side = owner.shape[0]
    return sum(int((one[1 + dx:1 + dx + side, 1 + dy:1 + dy + side, 1 + dz:1 + dz + side] & two).sum())
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))


@needs_gpu
def test_the_too_big_volume_splits_into_two_with_one_joint_between_and_the_halves_grow():
    state = new_universe(3, 10, seed=1, unevenness=0.0, margin=10)
    before = totals(state)
    parent = int((host_boxes(state)["quanta"] > 0).sum())
    step(state, Settings(split_across=9.0))
    boxes = host_boxes(state)
    counts = boxes["cells"]
    axis = int(np.argmax(counts))
    assert sorted(counts) == [3, 3, 5] and state.splits[0] == 1
    kinds = lattice_kinds(counts)
    assert np.array_equal(kinds.ravel()[boxes["owner"]], np.sign(boxes["quanta"]))
    places = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    places[1][axis], places[2][axis] = 3, 4
    low, joint, high = (int(np.ravel_multi_index(place, kinds.shape)) for place in places)
    assert int((kinds == 1).sum()) == 2 and kinds.ravel()[joint] == -1
    quanta, owner = boxes["quanta"], boxes["owner"]
    assert _touching(owner, quanta, low, high) == 0
    assert (quanta[owner == joint] < 0).all() and (owner == joint).sum() > 0
    after = totals(state)
    measured = stats(state, Settings(split_across=9.0))
    print(f"lattice {counts}; boxes low {(owner == low).sum()}, joint {(owner == joint).sum()}, high "
          f"{(owner == high).sum()}; torn {state.torn[0]}; amount {before['amount']} -> {after['amount']}; "
          f"wrap breaks {measured['wrap_breaks']}, web pieces {measured['web_pieces']}")
    assert after["amount"] == before["amount"]
    assert after["minus"] - before["minus"] == state.torn[0] == before["plus"] - after["plus"]
    assert measured["wrap_breaks"] == 0 and measured["web_pieces"] == 1
    for _ in range(75):
        step(state, Settings(split_across=0.0))
    found = cells(host_boxes(state))
    halves = int(found["boxes"][:found["volumes"]].sum())
    print(f"parent {parent} boxes; the two halves after 75 steps {halves} boxes; amount {totals(state)['amount']}")
    assert found["volumes"] == 2 and halves > parent
    assert totals(state)["amount"] == before["amount"]
