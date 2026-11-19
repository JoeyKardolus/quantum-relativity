"""(1) A lone full volume among lopsided thin neighbours: its own pull never moves it; with every rule on it still pours out."""
import numpy as np
import pytest

from physics.engine import step
from physics.engine.axiom_5_packing.logic.line import calm_block, line_of
from physics.engine.axiom_5_packing.logic.lumps import fuller
from physics.engine.axiom_5_packing.logic.pull import pull

CALM = 648_000


def _lopsided(full: int):
    """A line of five: the second volume holds `full` calm volumes, the others a quarter each; one thin
    neighbour stands on its one side behind a joint 4 times as packed, three on the other."""
    store = calm_block((1, 5, 1), {(0, place, 0): full if place == 1 else 0.25 for place in range(5)})
    cells = line_of(store)
    fuller(store, [cells[1]])
    return store, cells[::2]


def _speeds(store, volumes) -> np.ndarray:  # calm lengths a turn along the line
    return store.cell_flow.numpy()[volumes, 1] / store.cell_quanta.numpy()[volumes]


def test_the_pull_on_a_full_volume_is_the_same_whether_it_holds_100_or_1000_calm_volumes():
    gained, booked = [], []
    for full in (100, 1000):
        store, volumes = _lopsided(full)
        for _ in range(25):  # a turn of the pull alone
            pull(store)
            store.step += 1
        gained.append(_speeds(store, volumes))
        booked.append(int(store.cell_flow.numpy()[volumes, 1].sum()))
    assert abs(gained[0][1] - gained[1][1]) < 1e-3 and abs(gained[0][1]) < 0.02  # -0.016 and -0.016: the others' pull
    assert gained[0][0] > 18 and gained[0][2] < -10  # its neighbours fall toward it: 18.9 and -10.7 (ten times that at 1,000)
    assert abs(booked[0]) <= 25 * 5  # what one gains toward another, the other gains back: the flows sum to 0 (a rounding a step)
    assert booked[1] < -1_000_000  # not right: at 1,000 a neighbour is held at the top speed and its share is lost (-12,965,554)


@pytest.mark.xfail(strict=True, reason="(1) with every rule on the full volume does not keep its energy: 62 of 100 calm "
                   "volumes at 2 turns, 32 at 4, 21 at 8. Its own pull does not do it. Its neighbours fall toward it "
                   "at up to the top speed, friction holds their fluid back but hands their flow on, a little of the "
                   "full volume crosses, and that part a thin joint away pulls the rest after it. With the pull on "
                   "the full volume switched off (a comparison run) it holds 99.8 at 2 turns and 65 at 4")
def test_a_full_volume_among_lopsided_thin_neighbours_keeps_its_energy():
    store, volumes = _lopsided(100)
    for _ in range(50):
        step(store)
    assert store.cell_quanta.numpy()[volumes[1]] > 95 * CALM
