"""The universe grows from one volume: every joint made by a tear or the cut of one, each
between two volumes, the books exact at every step, the same seed the same growth."""
import numpy as np

from physics.engine import first_volume, step, three_cells
from physics.engine.axiom_6_limits.logic.tests.test_tear import _books, _joints_lie_between_two_volumes


def test_a_volume_no_bigger_than_the_limit_or_too_young_does_not_tear():
    store = three_cells(parts=(3, 0, 1))  # the emptier one is exactly twice a calm volume's size
    for _ in range(100):
        step(store)
    assert store.tears == 0 and store.cells == 3
    grows = first_volume()
    for _ in range(46):
        step(grows)
    assert grows.tears == 0  # a tear takes a turn of the whirl: 46.3 steps at 6 across
    step(grows)
    step(grows)
    assert grows.tears == 1


def _grown(seed: int = 1, steps: int = 190):
    """The universe from one volume through four turns of a whirl, checked at every step."""
    store = first_volume(seed=seed)
    for _ in range(steps):
        tears, (plus, minus), made = store.tears, _books(store), int(store.cell_made.numpy().sum())
        step(store)
        after_plus, after_minus = _books(store)
        assert plus + minus + int(store.cell_made.numpy().sum()) - made == after_plus + after_minus
        assert (after_plus < plus) == (store.tears > tears)  # plus only falls, and only by a tear
        _joints_lie_between_two_volumes(store)
    return store


def test_four_generations_grow_from_one_volume_with_exact_books_and_no_joint_but_by_a_tear():
    store = _grown()
    held = store.cell_quanta.numpy()[:store.cells]
    assert store.tears >= 7 and (held > 0).sum() == store.tears + 1 and store.cuts > 0
    print(f"{(held > 0).sum()} volumes, {(held < 0).sum()} joints after {store.step} steps")


def test_the_same_seed_gives_the_same_growth():
    first, again, other = _grown(1, 150), _grown(1, 150), _grown(2, 150)
    assert first.cells == again.cells > 3
    assert np.array_equal(first.box_quanta.numpy(), again.box_quanta.numpy())
    assert np.array_equal(first.joint_ends.numpy(), again.joint_ends.numpy())
    assert not np.array_equal(first.box_quanta.numpy()[:3], other.box_quanta.numpy()[:3])


def test_the_word_for_what_friction_was_once_booked_as_is_gone_from_the_code():
    from pathlib import Path
    code = Path(__file__).parents[4]  # physics/
    named = [str(file) for file in code.rglob("*.py")
             if "he" + "at" in file.read_text().lower() and "site-packages" not in str(file)]
    assert named == []
