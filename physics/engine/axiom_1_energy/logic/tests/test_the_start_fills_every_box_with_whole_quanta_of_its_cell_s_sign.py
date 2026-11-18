"""`start_energy`: whole quanta in every box, plus in volumes and minus in joints, from the seed alone."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_host
from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings
from physics.engine.axiom_1_energy.definitions.passes import (START_JOINT_A, START_JOINT_B, START_VOLUME_A,
                                                               START_VOLUME_B)
from physics.engine.axiom_1_energy.logic.start_energy import start_energy
from physics.engine.axiom_1_energy.logic.tests.conftest import all_blocks, small_grid
from physics.engine.axiom_1_energy.logic.uneven_field import uneven_field
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT

EVEN = replace(EnergySettings(), unevenness=0.0, inner_unevenness=0.0)


def _normal(seed: int, first_pass: int, second_pass: int, index: int) -> float:
    first = draw_uniform_host(seed, 0, first_pass, index)
    second = draw_uniform_host(seed, 0, second_pass, index)
    return math.sqrt(-2.0 * math.log(1.0 - first)) * math.cos(2.0 * math.pi * second)


def test_an_even_start_holds_exactly_the_start_numbers(grid):
    start_energy(grid, EVEN)
    quanta, kind = all_blocks(grid, "quanta"), grid.host("kind")
    assert (quanta[:grid.count][kind == ENERGY_VOLUME] == 3000).all()
    assert (quanta[:grid.count][kind == JOINT] == -100).all()
    assert not quanta[grid.count:].any()
    held = grid.host("cell_quanta")
    assert (held[kind == ENERGY_VOLUME] == 3000 * 216).all() and (held[kind == JOINT] == 100 * 216).all()


def test_every_box_is_whole_of_its_cell_s_sign_and_none_starts_empty(grid):
    start_energy(grid, replace(EnergySettings(), unevenness=0.3, inner_unevenness=0.5, joint_unevenness=0.4,
                               joint_quanta=1.0))
    quanta, kind = all_blocks(grid, "quanta")[:grid.count], grid.host("kind")
    assert quanta.dtype == np.int32
    assert (quanta[kind == ENERGY_VOLUME] >= 1).all() and (quanta[kind == JOINT] <= -1).all()
    assert np.array_equal(grid.host("cell_quanta"), np.abs(quanta.astype(np.int64)).sum(axis=(1, 2, 3)))


def test_each_box_holds_the_law_worked_by_hand(grid):
    settings = replace(EnergySettings(), unevenness=0.2, joint_unevenness=0.3)
    start_energy(grid, settings)
    kind = grid.host("kind")
    volumes = np.flatnonzero(kind == ENERGY_VOLUME)
    weight = np.exp(uneven_field(grid.side, grid.seed, settings.slope, settings.unevenness)).reshape(-1)
    for order, cell in ((3, int(volumes[3])), (None, int(np.flatnonzero(kind == JOINT)[10]))):
        block = grid.block("quanta", cell)
        if order is None:
            mean = settings.joint_quanta * math.exp(
                settings.joint_unevenness * _normal(grid.seed, START_JOINT_A, START_JOINT_B, cell))
        else:
            mean = settings.volume_quanta * weight[order] / weight.mean()
        for place in ((0, 0, 0), (2, 5, 1), (5, 5, 5)):
            box = cell * 216 + place[0] * 36 + place[1] * 6 + place[2]
            wanted = mean * math.exp(settings.inner_unevenness * _normal(grid.seed, START_VOLUME_A, START_VOLUME_B, box))
            assert abs(abs(int(block[place])) - wanted) <= 0.5 + 1e-6 * wanted


def test_the_mean_volume_box_holds_the_start_number(grid):
    start_energy(grid, replace(EnergySettings(), unevenness=0.3, inner_unevenness=0.0))
    held = grid.host("cell_quanta")[grid.host("kind") == ENERGY_VOLUME]
    assert abs(held.mean() / 216 - 3000.0) < 0.5


def test_the_same_seed_gives_the_same_start_and_it_writes_only_quanta():
    first, second, other = small_grid(seed=4), small_grid(seed=4), small_grid(seed=5)
    for grid in (first, second, other):
        start_energy(grid, EnergySettings())
    assert np.array_equal(all_blocks(first, "quanta"), all_blocks(second, "quanta"))
    assert not np.array_equal(all_blocks(first, "quanta"), all_blocks(other, "quanta"))
    assert not all_blocks(first, "phase").any()
    assert not first.host("length").any() and not first.host("moved_inside").any()
