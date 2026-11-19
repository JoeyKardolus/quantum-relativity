"""The same limit a second time: out of a volume past it nothing leaves, by the crossing or by the leak."""
from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.definitions.store import put
from physics.engine.axiom_4_crossing.logic.leak import leak
from physics.engine.axiom_4_crossing.logic.tests.test_cross import FAR, A, B, _still
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP


def _gained(way: int, deep=None) -> int:
    """What A gains in 30 steps, its fluid moving toward B (`way` 1) or away; `deep` is held past the limit."""
    store = _still((2, -2, 2), a_phase=-way * 0.3 * FAR)
    for _ in range(30):
        if deep is not None:
            put(store.cell_pull, deep, 2 * DEEP)
        step(store)
    return int(store.cell_quanta.numpy()[A]) - 648_000


def test_nothing_crosses_out_of_a_volume_past_the_limit_though_it_still_takes_in():
    assert _gained(1) < -1000 and _gained(1, deep=B) < -1000  # A gives; into a deep B too
    assert _gained(1, deep=A) >= 0 and _gained(-1, deep=A) > 1000  # out of a deep A nothing; in, yes


def test_a_volume_past_the_limit_does_not_leak():
    left = []
    for pull in (0.999 * DEEP, DEEP):
        store = first_volume(layers=1)
        put(store.cell_pull, store.side ** 3 // 2, pull)
        for _ in range(25):
            leak(store)
        left.append(2 ** 30 - int(store.cell_quanta.numpy()[store.side ** 3 // 2]))
    assert left[0] > 600_000 and left[1] == 0  # a calm volume's worth in a turn; past the limit: nothing
