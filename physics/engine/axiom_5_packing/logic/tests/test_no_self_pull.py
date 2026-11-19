"""(1) A lone full volume among lopsided thin neighbours: its own pull never moves it, and with every rule on it keeps its energy."""
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
    assert abs(booked[1]) <= 25 * 5  # at 1,000 too: a neighbour past the top speed keeps its flow on the books


def test_a_full_volume_among_lopsided_thin_neighbours_keeps_its_energy():
    store, volumes = _lopsided(100)
    for _ in range(50):
        step(store)
    held = store.cell_quanta.numpy()[volumes[1]] / CALM
    print(f"the full volume holds {held:.2f} of 100 calm volumes after 2 turns")
    assert held > 95  # it poured out before (62 at 2 turns): the neighbours' flow only pressed, and its own crossed
