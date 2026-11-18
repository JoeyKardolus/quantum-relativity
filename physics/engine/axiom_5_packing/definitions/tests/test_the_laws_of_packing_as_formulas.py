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
    out[2] = pull_source(wp.int32(given[4]), given[2], given[5])
    out[3] = wave_weight(wp.float64(1.0), wp.float64(0.25), given[6], given[7])
    out[4] = wave_step(given[8], given[9], given[10], wp.float64(1.0), wp.float64(0.25), given[11])
    out[5] = wave_rest(given[10], wp.float64(1.0), wp.float64(0.25), given[11])
    out[6] = way_press_law(given[12], given[13], given[14], given[15], given[16], given[17], given[18])
    out[7] = push_law(wp.int64(given[19]), given[20], given[21])
    out[8] = wp.float64(squeeze_law(given[22], given[23]))


def _run(**changed) -> np.ndarray:
    given = {"quanta": 648000, "length": 10.0, "packing": 6480.0, "clock_packing": 6480.0, "kind": ENERGY_VOLUME,
             "strength": 1e-7, "own": 10.0, "other": 300.0, "pull": -1.0, "pull_old": -1.5, "heard": 0.02,
             "source": 0.08, "quanta_a": 3000.0, "far_a": 2.0, "reach_a": 10.0, "quanta_b": 3000.0, "far_b": 2.0,
             "reach_b": 10.0, "pull_joint": 1.0, "joint_quanta": 21600, "joint_length": 300.0, "push": 2e-6,
             "press": 5.0, "pushed": 2.0}
    given.update(changed)
    out = wp.zeros(9, dtype=wp.float64, device="cpu")
    wp.launch(_laws, dim=1, inputs=[wp.array(list(given.values()), dtype=wp.float64, device="cpu")], outputs=[out],
              device="cpu")
    return out.numpy()


def test_more_packed_is_slower_and_only_plus_energy_is_a_source_that_grows_with_packing():
    out = _run()
    assert out[0] == 648.0 and out[1] == 0.5
    assert _run(packing=0.0)[1] == 1.0 and 0.0 < _run(packing=6.48e12)[1] < 1e-8
    assert math.isclose(out[2], 4.0 * math.pi * 1e-7 * 6480.0)
    assert _run(packing=12960.0)[2] == 2.0 * out[2]
    assert _run(kind=JOINT)[2] == 0.0 and _run(kind=0)[2] == 0.0


def test_the_wave_step_has_no_clock_in_it_and_a_field_at_rest_stays_at_rest():
    out = _run()
    assert math.isclose(out[3], 6.0 * (0.25 / 155.0) ** 2)
    assert _run(own=0.1, other=0.1)[3] == 1.0 and _run(other=float("inf"))[3] == 0.0
    assert math.isclose(out[5], 0.02 - 0.0625 * 0.08)
    assert math.isclose(out[4], 2.0 * -1.0 + 1.5 + out[5])
    assert _run(pull_old=-1.0, heard=0.0625 * 0.08)[4] == -1.0


def test_two_volumes_falling_together_down_one_slope_press_nothing_whatever_each_holds():
    for quanta_a, quanta_b in ((3000.0, 3000.0), (3000.0, 3300.0), (3300.0, 3000.0), (1.0, 1e9)):
        assert _run(quanta_a=quanta_a, quanta_b=quanta_b, far_a=2.0, far_b=0.0)[6] == 0.0
    assert _run()[6] == 1500.0 * 0.2                               # both pulled toward the joint: pressed
    assert _run(far_a=0.0, far_b=0.0)[6] == -1500.0 * 0.2          # both pulled away: drawn apart
    assert _run(quanta_b=0.0)[6] == 0.0 and _run(reach_a=float("inf"), far_b=1.0)[6] == 0.0
    assert _run(quanta_a=6000.0, quanta_b=6000.0)[6] == 2.0 * _run()[6]


def test_squeezed_minus_pushes_back_steeply_with_the_fourth_power_of_its_quanta():
    out = _run()
    assert math.isclose(out[7], 2e-6 * 72.0 ** 2)
    double = _run(joint_quanta=43200, joint_length=150.0)[7]       # twice the quanta is half the length
    assert math.isclose(double, 16.0 * out[7])
    assert _run(joint_quanta=0)[7] == 0.0
    assert out[8] == 2.5 and _run(press=-5.0)[8] == 0.0 and _run(pushed=0.0)[8] == 0.0
