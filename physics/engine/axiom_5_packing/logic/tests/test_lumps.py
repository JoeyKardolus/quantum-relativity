"""Staged checks in the grown patch, two volumes out from the first; each gap is a strict xfail with its measured number.

The same checks hold in a calm line of cells (`test_carried.py`); here the first volume has
emptied into its neighbours and energy streams outward, and a staged region is torn along.
"""
import numpy as np
import pytest

from physics.engine import step
from physics.engine.axiom_5_packing.logic.lumps import FULLER, fuller, moving, region
from physics.engine.axiom_5_packing.logic.staging import copy_of
from physics.engine.axiom_6_limits.logic.tests.test_grow import _checked_step

OUT = np.arange(-3, 4)  # the places of the row through the middle


def _stands(store, out: tuple) -> list:  # those of the region `out` volumes from the middle that stand
    return [cell for cell in region(store, out) if cell in store.own_cell.numpy()[:store.count]]


def _row(store) -> np.ndarray:  # the quanta along the row through the middle, in calm volumes
    cells = [region(store, (0, out, 0))[0] for out in OUT]
    return store.cell_quanta.numpy()[cells] / 648_000


def _after(grown_patch, stage, steps: int) -> tuple:
    """The rows of the staged patch and of the patch left alone, both `steps` steps on."""
    staged, plain = copy_of(grown_patch), copy_of(grown_patch)
    stage(staged)
    first = _row(staged) - _row(plain)
    for _ in range(steps):
        step(staged)
        step(plain)
    return first, _row(staged) - _row(plain)


def _middle(more: np.ndarray, where=slice(None)) -> float:
    """The middle of the energy a row holds more than the patch left alone, in volumes along it."""
    weight = np.clip(more, 0, None)[where]
    return float((weight * OUT[where]).sum() / weight.sum())


def test_staging_makes_a_region_four_times_fuller_and_sets_it_going_with_exact_books(patch):
    cells = _stands(patch, (0, 2, 0))
    before = patch.cell_quanta.numpy()[cells]
    was = patch.cell_flow.numpy()[cells]
    fuller(patch, cells)
    moving(patch, cells, 1, 2.0)
    assert (patch.cell_quanta.numpy()[cells] == FULLER * before).all()
    flow = patch.cell_flow.numpy()[cells]
    assert (flow[:, 1] == 2 * FULLER * before).all() and (flow[:, [0, 2]] == was[:, [0, 2]]).all()
    for _ in range(5):
        _checked_step(patch)  # the books stay exact with the pull on


@pytest.mark.xfail(strict=True, reason="(a) in the grown patch a region made 4 times fuller does not gain and no longer "
                   "holds: 0.56 of what it was given after a turn (0.95 before the fluid carried its flow); the "
                   "patch is streaming outward from the emptied first volume and the region goes with it")
def test_a_fuller_region_gains_from_its_thinner_surroundings(grown_patch):
    cells = _stands(grown_patch, (0, 2, 0))
    first, more = _after(grown_patch, lambda store: fuller(store, cells), 25)
    assert more[4:7].sum() > first[4:7].sum()


@pytest.mark.xfail(strict=True, reason="(b) in the grown patch two fuller regions either side of the first volume do "
                   "not draw together: 2.37 volumes apart, 2.95 after a turn; the patch is lopsided (3 calm "
                   "volumes one side, 915 the other) and streaming outward")
def test_two_fuller_regions_draw_together(grown_patch):
    stage = lambda store: [fuller(store, _stands(store, (0, way, 0))) for way in (-2, 2)]  # noqa: E731
    first, more = _after(grown_patch, stage, 25)
    apart = lambda row: _middle(row, slice(4, 7)) - _middle(row, slice(0, 3))  # noqa: E731
    assert apart(more) < apart(first)
