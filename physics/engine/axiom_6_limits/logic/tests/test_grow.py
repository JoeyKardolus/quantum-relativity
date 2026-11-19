"""The universe grows from one volume, layer by layer, a joint wherever two volumes share a side, the books exact at every step."""
from functools import lru_cache

import numpy as np
import pytest

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.definitions.places import steps_of


def _cells(store) -> tuple:
    """Every place's quanta, the own volumes and the own joints."""
    held, own = store.cell_quanta.numpy().astype(np.int64), store.own_cell.numpy()[:store.count]
    boxes = store.box_quanta.numpy()[:store.count].astype(np.int64).sum(axis=1)
    assert (boxes == held[own]).all()  # each cell's total is its boxes'
    return held, own[held[own] > 0], own[held[own] < 0]


def _layers(store, volumes: np.ndarray) -> np.ndarray:
    """How many layers out from the first volume each volume lies."""
    return np.abs(steps_of(volumes, store.side) - store.side // 2).max(axis=1) // 2


def _checked_step(store) -> tuple:
    """One step, its books and its pattern checked; the volumes and joints after it."""
    before, made = _cells(store)[0], int(store.cell_made.numpy().sum())
    step(store)
    held, volumes, joints = _cells(store)
    made = int(store.cell_made.numpy().sum()) - made
    flipped = before[before > 0].sum() - held[held > 0].sum()
    assert flipped >= 0 and held[held < 0].sum() == before[before < 0].sum() - flipped - made
    axis = np.argmax(steps_of(joints, store.side) % 2, axis=1)  # a joint's one odd step
    assert (steps_of(joints, store.side) % 2).sum() == len(joints) == (held < 0).sum()
    stands = np.isin(np.arange(held.size), volumes)
    apart = store.side ** (2 - axis)
    assert stands[joints - apart].all() and stands[joints + apart].all()  # between two volumes
    pairs = sum(int((stands[volumes] & stands[volumes + 2 * store.side ** way]).sum()) for way in range(3))
    assert pairs == len(joints)  # a joint wherever two volumes share a side, and nowhere else
    return volumes, joints


@lru_cache
def _grown(layers: int = 2):
    """A universe grown until a volume stands `layers` layers out, checked at every step; made once."""
    store, seen = first_volume(layers=layers + 1), {}
    while store.step < 700:
        volumes, joints = _checked_step(store)
        seen["least"] = min(seen.get("least", 2 ** 31), int(store.cell_quanta.numpy()[volumes].min()))
        if len(volumes) == 27 and "layer" not in seen:
            seen["layer"] = (store.step, volumes, joints, _layers(store, volumes))
        if _layers(store, volumes).max() == layers:
            return store, seen
    raise AssertionError("no second layer in 700 steps")


grown = pytest.fixture(scope="module")(lambda: _grown())


def test_a_lone_volume_sheds_a_full_layer_of_26_volumes_with_a_joint_wherever_two_share_a_side(grown):
    store, seen = grown
    at, volumes, joints, layers = seen["layer"]
    middle = store.side ** 3 // 2
    assert sorted(layers) == [0] + [1] * 26 and len(joints) == 54
    assert sum(abs(joint - middle) in (1, store.side, store.side ** 2) for joint in joints) == 6
    print(f"the first layer stands whole at step {at}: {at * store.turn_per_step / 2 / np.pi:.1f} turns")


def test_two_layers_grow_with_exact_books_and_the_outermost_cells_are_volumes(grown):
    store, _seen = grown
    held, volumes, joints = _cells(store)
    reach = np.abs(steps_of(np.concatenate([volumes, joints]), store.side) - store.side // 2).max()
    assert reach == 4 and len(volumes) > 27  # four steps out: the second layer, at step 491 (19.7 turns)
    assert held[held > 0].sum() - held[held < 0].sum() == 2 ** 30 + store.cell_made.numpy().sum()
