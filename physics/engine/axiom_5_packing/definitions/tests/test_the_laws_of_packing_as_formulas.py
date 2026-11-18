"""Axiom 5's laws on plain numbers: packing, the clock, the source, the wave, the press and the push."""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_5_packing.definitions.clock_rate import clock_rate
from physics.engine.axiom_5_packing.definitions.packing_law import packing_law
from physics.engine.axiom_5_packing.definitions.press_law import way_press_law
from physics.engine.axiom_5_packing.definitions.pull_source import pull_source
from physics.engine.axiom_5_packing.definitions.pull_wave_law import wave_rest, wave_step, wave_weight
from physics.engine.axiom_5_packing.definitions.push_law import push_law, squeeze_law

wp.config.quiet = True


@wp.kernel
def _laws(given: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    out[0] = packing_law(wp.int64(given[0]), given[1])
    out[1] = wp.float64(clock_rate(given[2], given[3]))
    out[2] = pull_source(wp.int32(given[4]), wp.int64(given[0]), given[24], given[5])
    out[3] = wave_weight(wp.float64(1.0), wp.float64(0.25), (given[6] + given[7]) / wp.float64(2.0))
    out[4] = wave_step(given[8], given[9], given[25], given[10], given[24], given[11])
    out[5] = wave_rest(given[10], given[24], given[11])
    out[6] = way_press_law(given[12], given[13], given[14], given[15], given[16], given[17], given[18])
    out[7] = push_law(wp.int64(given[19]), given[20], given[21])
    out[8] = wp.float64(squeeze_law(given[22], given[23]))


def _run(**changed) -> np.ndarray:
    given = {"quanta": 648000, "length": 10.0, "packing": 6480.0, "clock_packing": 6480.0, "kind": ENERGY_VOLUME,
             "strength": 1e-7, "own": 10.0, "other": 300.0, "pull": -1.0, "pull_old": -1.5, "heard": 0.02,
             "source": 0.08, "quanta_a": 3000.0, "pull_a": 2.0, "reach_a": 10.0, "quanta_b": 3000.0, "pull_b": 2.0,
             "reach_b": 10.0, "pull_joint": 1.0, "joint_quanta": 21600, "joint_length": 300.0, "push": 2e-6,
             "press": 5.0, "pushed": 2.0, "reach": 1.5, "weight": 0.5}
    given.update(changed)
    out = wp.zeros(9, dtype=wp.float64, device="cpu")
    wp.launch(_laws, dim=1, inputs=[wp.array(list(given.values()), dtype=wp.float64, device="cpu")], outputs=[out],
              device="cpu")
    return out.numpy()


def test_more_packed_is_slower_and_only_plus_energy_is_a_source_that_grows_with_what_it_holds():
    out = _run()
    assert out[0] == 648.0 and out[1] == 0.5
    assert _run(packing=0.0)[1] == 1.0 and 0.0 < _run(packing=6.48e12)[1] < 1e-8
    assert math.isclose(out[2], 4.0 * math.pi * 1e-7 * 648000.0 / 1.5 ** 3)
    assert math.isclose(_run(reach=0.75)[2], 8.0 * out[2])          # the same energy in half the room: more packed
    assert _run(quanta=1296000)[2] == 2.0 * out[2]                  # twice the energy pulls twice as hard ...
    assert _run(packing=12960.0)[2] == out[2]                       # ... and is pulled twice as hard: no fourth power
    assert _run(kind=JOINT)[2] == 0.0 and _run(kind=0)[2] == 0.0 and _run(quanta=0)[2] == 0.0
    assert _run(reach=0.0)[2] == 0.0 and _run(reach=float("inf"))[2] == 0.0


def test_the_wave_step_has_no_clock_in_it_and_a_field_at_rest_stays_at_rest():
    out = _run()
    assert math.isclose(out[3], (0.25 / 155.0) ** 2)
    assert _run(own=0.1, other=0.1)[3] == 1.0 and _run(other=float("inf"))[3] == 0.0
    assert math.isclose(out[5], 0.02 - 1.5 ** 2 * 0.08 / 6.0)
    assert math.isclose(out[4], 2.0 * -1.0 + 1.5 + 0.5 * out[5])
    assert _run(pull_old=-1.0, heard=1.5 ** 2 * 0.08 / 6.0)[4] == -1.0


def test_two_volumes_on_one_even_slope_press_nothing_and_two_that_pull_at_the_joint_press_it():
    for quanta_a, quanta_b in ((3000.0, 3000.0), (3000.0, 3300.0), (3300.0, 3000.0), (1.0, 1e9)):
        assert _run(quanta_a=quanta_a, quanta_b=quanta_b, pull_a=2.0, pull_b=0.0)[6] == 0.0
    assert _run(pull_a=0.0, pull_b=0.0)[6] == 1500.0 * 0.2         # both volumes lie deeper than the joint: pressed
    assert _run()[6] == -1500.0 * 0.2                              # the joint lies deeper than both: drawn apart
    assert _run(quanta_b=0.0)[6] == 0.0 and _run(reach_a=float("inf"), pull_b=1.0)[6] == 0.0
    assert _run(quanta_a=6000.0, quanta_b=6000.0)[6] == 2.0 * _run()[6]


def test_squeezed_minus_pushes_back_steeply_with_the_fourth_power_of_its_quanta():
    out = _run()
    assert math.isclose(out[7], 2e-6 * 72.0 ** 2)
    double = _run(joint_quanta=43200, joint_length=150.0)[7]       # twice the quanta is half the length
    assert math.isclose(double, 16.0 * out[7])
    assert _run(joint_quanta=0)[7] == 0.0
    assert out[8] == 2.5 and _run(press=-5.0)[8] == 0.0 and _run(pushed=0.0)[8] == 0.0
