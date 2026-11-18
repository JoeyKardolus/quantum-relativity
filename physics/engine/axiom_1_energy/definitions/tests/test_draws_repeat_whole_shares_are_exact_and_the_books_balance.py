"""The draws, whole shares, the book law, the arrays and the settings of axiom 1."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_1_energy.definitions import passes
from physics.engine.axiom_1_energy.definitions.book_law import book_law
from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform, draw_uniform_host, draw_uniform_rows
from physics.engine.axiom_1_energy.definitions.energy_arrays import ARRAYS, HOST
from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings
from physics.engine.axiom_1_energy.definitions.whole_shares import whole_shares

wp.config.quiet = True


@wp.kernel
def _draw(seed: wp.uint32, step: wp.uint32, pass_id: wp.uint32, draws: wp.array(dtype=wp.float64)):
    index = wp.tid()
    draws[index] = draw_uniform(seed, step, pass_id, wp.uint64(index))


def _draws(seed: int, step: int, pass_id: int, number: int = 20000) -> np.ndarray:
    draws = wp.zeros(number, dtype=wp.float64, device="cpu")
    wp.launch(_draw, dim=number, inputs=[seed, step, pass_id, draws], device="cpu")
    return draws.numpy()


def test_a_draw_is_the_same_in_a_kernel_and_on_the_host_and_repeats():
    draws = _draws(1, 7, passes.FLOW_ROUND)
    assert draws[:50].tolist() == [draw_uniform_host(1, 7, passes.FLOW_ROUND, index) for index in range(50)]
    assert np.array_equal(draws, _draws(1, 7, passes.FLOW_ROUND))
    assert draw_uniform_host(1, 7, 10, 2 ** 40 + 5) != draw_uniform_host(1, 7, 10, 5)     # a 64-bit index counts


def test_draws_fill_zero_to_one_evenly_and_differ_by_seed_step_and_pass():
    draws = _draws(1, 7, passes.FLOW_ROUND)
    assert draws.min() >= 0.0 and draws.max() < 1.0
    assert abs(draws.mean() - 0.5) < 0.01 and abs(draws.var() - 1.0 / 12.0) < 0.005
    assert (np.histogram(draws, bins=10, range=(0.0, 1.0))[0] > 1800).all()
    for other in (_draws(2, 7, passes.FLOW_ROUND), _draws(1, 8, passes.FLOW_ROUND), _draws(1, 7, passes.WAY_ROUND)):
        assert abs(np.corrcoef(draws, other)[0, 1]) < 0.03
    assert abs(np.corrcoef(draws[:-1], draws[1:])[0, 1]) < 0.03


def test_every_use_of_a_draw_has_its_own_pass():
    numbers = {name: value for name, value in vars(passes).items() if name.isupper()}
    assert numbers == {"START_VOLUME_A": 1, "START_VOLUME_B": 2, "START_JOINT_A": 3, "START_JOINT_B": 4,
                       "START_SWIRL_AXIS": 5, "START_SWIRL_SENSE": 6, "START_FIELD_A": 7,
                       "START_FIELD_B": 8, "FLOW_ROUND": 10, "WAY_ROUND": 20,
                       "WAY_FRICTION": 21, "MAKE_ROUND": 30, "SPLIT_SPEED": 40,
                       "SPLIT_POINTER": 41, "SETTLE_DEAL": 42, "SPLIT_CARRY": 43, "SPLIT_DEAL": 44,
                       "SPLIT_COOL": 45}
    used = [value for name, value in numbers.items() if name != "FLOW_ROUND"]
    used += [passes.FLOW_ROUND + axis * 2 + parity for axis in range(3) for parity in range(2)]
    assert len(set(used)) == len(used)


def test_hand_worked_whole_shares():
    assert whole_shares(10, [1, 1, 1]) == [4, 3, 3]                 # the rest to the lowest index
    assert whole_shares(7, [3, 1]) == [5, 2]                        # 5.25 and 1.75: the larger remainder gets it
    assert whole_shares(5, [0, 0, 0]) == [2, 2, 1]                  # all weights 0: evenly
    assert whole_shares(0, [4, 9]) == [0, 0]
    assert whole_shares(6, [0, 5, 0, 5]) == [0, 3, 0, 3]            # a box holding nothing gives nothing
    assert whole_shares(0, []) == []
    for wrong in ((-1, [1]), (3, []), (3, [1, -1]), (3, [0.5, 1])):
        with pytest.raises(ValueError):
            whole_shares(*wrong)


def test_whole_shares_are_exact_and_never_ask_a_box_for_more_than_it_holds():
    rng = np.random.default_rng(5)
    for _ in range(200):
        holds = rng.integers(0, 4000, 16)
        total = int(rng.integers(0, int(holds.sum()) + 1))
        shares = whole_shares(total, holds)
        assert sum(shares) == total and all(0 <= share <= held for share, held in zip(shares, holds))
    huge = whole_shares(10 ** 30 + 1, [10 ** 20, 2 * 10 ** 20])
    assert sum(huge) == 10 ** 30 + 1 and huge[1] - 2 * huge[0] in (-2, -1, 0, 1, 2)


def test_hand_worked_books():
    start = {"plus": 1000, "minus": 200, "amount": 1200}
    still = {**start, "made_total": 0, "taken_joint": 0, "taken_volumes": 0, "short_quanta": 0, "torn": 0}
    assert book_law(start, still) == {"amount": 0, "plus": 0, "minus": 0, "taken": 0}
    # joints made 50; a collapse took a joint of 30 and 30 off its volumes; a tear turned 100 plus into minus
    now = {"plus": 1000 - 30 - 100, "minus": 200 + 50 - 30 + 100, "made_total": 50, "taken_joint": 30,
           "taken_volumes": 30, "short_quanta": 0, "torn": 100}
    now["amount"] = now["plus"] + now["minus"]
    assert now["amount"] == 1200 + 50 - 30 - 30
    assert book_law(start, now) == {"amount": 0, "plus": 0, "minus": 0, "taken": 0}
    assert book_law(start, {**now, "plus": now["plus"] - 1, "amount": now["amount"] - 1}) == \
        {"amount": -1, "plus": -1, "minus": 0, "taken": 0}          # one quantum gone unbooked shows
    # a collapse that took 5 fewer off the rest than its joint held, with nothing booked as short, shows
    short = {**now, "plus": now["plus"] + 5, "amount": now["amount"] + 5, "taken_volumes": 25}
    assert book_law(start, short) == {"amount": 0, "plus": 0, "minus": 0, "taken": -5}
    assert book_law(start, {**short, "short_quanta": 5})["taken"] == 0


def test_arrays_and_host_counts():
    assert ARRAYS == {"quanta": ("int32", "box", False), "cell_quanta": ("int64", "cell", False),
                      "moved_inside": ("int64", "cell", True), "cut_moves": ("int32", "cell", True)}
    assert HOST == {"made_total": 0, "taken_joint": 0, "taken_volumes": 0, "short_takes": 0, "short_quanta": 0,
                    "torn": 0, "start_books": None}


def test_settings_defaults_and_what_they_refuse():
    settings = EnergySettings()
    assert (settings.step_time, settings.volume_quanta, settings.joint_quanta) == (0.25, 3000, 100)
    assert not hasattr(settings, "fluid_pressure")
    assert (settings.unevenness, settings.slope, settings.inner_unevenness, settings.joint_unevenness) == \
        (0.1, -3.0, 0.1, 0.0)
    for name in ("step_time", "volume_quanta", "joint_quanta"):
        with pytest.raises(ValueError):
            EnergySettings(**{name: 0.0})
    with pytest.raises(ValueError):
        EnergySettings(unevenness=-0.1)
    assert EnergySettings(unevenness=0.0, slope=2.0).unevenness == 0.0      # an even start is allowed
    with pytest.raises(TypeError):
        EnergySettings(crossing_rate=1.0)


def test_a_whole_array_of_draws_is_the_same_draws_one_by_one():
    indexes = [0, 1, 215, 2 ** 32 + 7, 2 ** 63 + 5]
    rows = draw_uniform_rows(11, 3, 41, np.array(indexes, dtype=np.uint64))
    assert rows.tolist() == [draw_uniform_host(11, 3, 41, index) for index in indexes]
