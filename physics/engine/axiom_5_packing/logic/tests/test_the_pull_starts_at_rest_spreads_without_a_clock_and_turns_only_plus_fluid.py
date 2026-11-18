"""The pull on a real small universe: at rest from the start, deeper at more packed energy, felt only by plus."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics import engine
from physics.engine.axiom_1_energy.logic.clear_ledgers import clear_ledgers
from physics.engine.axiom_5_packing.logic.pull_fluid import pull_fluid
from physics.engine.axiom_5_packing.logic.spread_pull import spread_pull
from physics.engine.axiom_5_packing.logic.tick_clocks import tick_clocks

wp.config.quiet = True


def _universe(side: int = 8, **changed):
    settings = engine.Settings(**changed)
    return engine.new_universe(settings, cells_per_side=side, seed=1, device="cpu"), settings


def test_the_start_s_pull_stands_still_so_no_wave_is_switched_on():
    state, settings = _universe()
    pull = state.host("pull").astype(np.float64)
    spread = pull.max() - pull.min()
    assert spread > 0.0 and abs(pull.mean()) < 1e-4 * spread
    assert np.array_equal(state.host("pull"), state.host("pull_old"))
    for _ in range(20):
        clear_ledgers(state)
        spread_pull(state, settings)
    after = state.host("pull")
    moved = np.abs((after - after.mean()) - (pull - pull.mean())).max()      # the level all cells share sinks
    assert moved < 1e-3 * spread


def test_at_rest_more_packed_volumes_sit_deeper_and_every_volume_sits_below_its_own_joints():
    state, settings = _universe()
    kind, pull, packing = state.host("kind"), state.host("pull").astype(np.float64), state.host("packing")
    volumes = kind == engine.ENERGY_VOLUME
    assert np.corrcoef(packing[volumes], pull[volumes])[0, 1] < -0.5
    ways = engine.read_ways(state)
    assert (pull[ways[:, 3]] > np.minimum(pull[ways[:, 1]], pull[ways[:, 2]])).mean() > 0.95


def test_a_slow_clock_does_not_make_a_packed_cell_pull_less():
    state, settings = _universe()
    before = state.host("pull").copy()
    state.rate.fill_(1e-9)
    clear_ledgers(state)
    spread_pull(state, settings)
    slow = state.host("pull").copy()
    state.pull.assign(np.pad(before, (0, state.capacity - before.size)))
    state.pull_old.assign(np.pad(before, (0, state.capacity - before.size)))
    state.rate.fill_(1.0)
    clear_ledgers(state)
    spread_pull(state, settings)
    assert np.array_equal(slow, state.host("pull"))
    made = state.host("pull_made")
    assert (made[state.host("kind") == engine.ENERGY_VOLUME] > 0).all()
    assert not made[state.host("kind") == engine.JOINT].any()


def _rise(state, volume: int, step: tuple) -> float:
    """The pull step from a volume to its neighbour in one direction: the pull at that side over its middle."""
    pull = state.host("pull")
    code = (step[0] + 1) * 9 + (step[1] + 1) * 3 + step[2] + 1
    joint = next(one for one, there in state.neighbours(volume) if there == code)
    return float(pull[joint] - pull[volume])


def test_the_pull_turns_only_plus_fluid_across_the_whole_cell_and_adds_up_on_a_slow_clock():
    state, settings = _universe(side=4)
    kind = state.host("kind")
    state.phase.zero_()
    clear_ledgers(state)
    pull_fluid(state, settings)
    phase = state.phase.numpy()
    felt = state.host("pull_felt")
    assert (felt[kind == engine.ENERGY_VOLUME] > 0).all() and not felt[kind == engine.JOINT].any()
    assert not phase[:state.count][kind == engine.JOINT].any()
    volume = int(np.flatnonzero(kind == engine.ENERGY_VOLUME)[0])
    tau = 0.25 * float(state.host("rate")[volume])
    own = float(state.host("pull")[volume])
    # box (5, 2, 2) stands at the +x side and a fifth of the way toward -y and toward -z
    whole = 0.25 * own
    expected = whole + tau * (0.64 * _rise(state, volume, (1, 0, 0)) + 0.16 * _rise(state, volume, (1, -1, 0))
                      + 0.16 * _rise(state, volume, (1, 0, -1)) + 0.04 * _rise(state, volume, (1, -1, -1)))
    assert np.isclose(phase[volume, 5, 2, 2], expected, rtol=1e-9)
    # box (3, 2, 2) stands a fifth of the way toward +x: the pull slopes across the inside too, it is no step at the skin
    inner = whole + tau * (0.2 * 0.64 * _rise(state, volume, (1, 0, 0)) + 0.8 * 0.16 * _rise(state, volume, (0, -1, 0))
                   + 0.8 * 0.16 * _rise(state, volume, (0, 0, -1)) + 0.8 * 0.04 * _rise(state, volume, (0, -1, -1))
                   + 0.2 * 0.16 * _rise(state, volume, (1, -1, 0)) + 0.2 * 0.16 * _rise(state, volume, (1, 0, -1))
                   + 0.2 * 0.04 * _rise(state, volume, (1, -1, -1)))
    assert np.isclose(phase[volume, 3, 2, 2], inner, rtol=1e-9) and 0.0 < abs(inner - whole) < abs(expected - whole)
    assert float(felt[volume]) == pytest.approx(np.abs(phase[volume] - whole).max(), rel=1e-6)
    state.phase.zero_()
    state.rate.fill_(1e-9)
    for _ in range(1000):
        pull_fluid(state, settings)
    slow = state.phase.numpy()[volume]
    sloped = float(np.angle(np.exp(1j * (slow[5, 2, 2] - slow[3, 2, 2]))))      # the whole turn is the same for both
    assert np.isclose(sloped, 1000 * (expected - inner) / tau * 0.25 * 1e-9, rtol=1e-4) and sloped != 0.0


def test_an_emptied_neighbour_is_empty_space_and_its_old_pull_turns_nobody():
    state, settings = _universe(side=4)
    kind = state.host("kind")
    volume = int(np.flatnonzero(kind == engine.ENERGY_VOLUME)[0])
    joint = next(one for one, there in state.neighbours(volume) if there == 22)
    state.set_cell("cell_quanta", joint, 0)
    state.set_cell("pull", joint, 1.0e6)
    state.phase.zero_()
    clear_ledgers(state)
    pull_fluid(state, settings)
    phase = state.phase.numpy()[volume]
    assert np.abs(phase).max() < 10.0
    assert phase[5, 2, 2] == pytest.approx(0.25 * float(state.host("pull")[volume]) + 0.25 * float(state.host("rate")[volume]) * (
        0.16 * _rise(state, volume, (1, -1, 0)) + 0.16 * _rise(state, volume, (1, 0, -1))
        + 0.04 * _rise(state, volume, (1, -1, -1))), rel=1e-9)


def test_a_clock_counts_the_local_time_a_cell_lived_and_an_empty_cell_s_clock_stands():
    state, settings = _universe(side=4)
    state.set_block("quanta", 3, np.zeros((6, 6, 6), dtype=np.int32))
    state.set_cell("cell_quanta", 3, 0)
    for _ in range(8):
        tick_clocks(state, settings)
    clock, rate = state.host("clock"), state.host("rate").astype(np.float64)
    assert clock[3] == 0.0
    assert np.allclose(np.delete(clock, 3), np.delete(8 * 0.25 * rate, 3), rtol=1e-12)
