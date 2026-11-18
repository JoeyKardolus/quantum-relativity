"""Axiom 6's laws on plain numbers: what a squeezed joint makes, when it collapses, when a volume tears."""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from physics.engine.axiom_6_limits.definitions.collapse_law import collapse_law
from physics.engine.axiom_6_limits.definitions.make_law import make_law
from physics.engine.axiom_6_limits.definitions.tear_law import tear_law
from physics.engine.axiom_6_limits.definitions.tear_map import tear_map

wp.config.quiet = True


@wp.kernel
def _laws(given: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    out[0] = make_law(wp.int64(given[0]), given[1], given[2], given[3])
    if collapse_law(wp.int64(given[4]), wp.int64(given[5]), given[6]):
        out[1] = wp.float64(1.0)
    if tear_law(given[7], wp.int64(given[8]), wp.int64(given[9]), wp.int64(given[10]), given[11]):
        out[2] = wp.float64(1.0)


def _run(quanta=21600, squeeze=16.0, rate=1.0, tau=0.25, holds=64800, fresh=21600, thin=3.0, length=25.0,
         pieces=(5, 5, 5), tear_length=20.0) -> list:
    out = wp.zeros(3, dtype=wp.float64, device="cpu")
    given = [quanta, squeeze, rate, tau, holds, fresh, thin, length, *pieces, tear_length]
    wp.launch(_laws, dim=1, inputs=[wp.array(given, dtype=wp.float64, device="cpu")], outputs=[out], device="cpu")
    return out.numpy().tolist()


def test_a_joint_makes_only_while_squeezed_toward_the_level_at_its_rate_on_its_own_clock():
    level = 21600 * (16.0 ** 0.25 - 1.0)
    assert math.isclose(_run()[0], level * (1.0 - math.exp(-0.25)))
    assert _run(squeeze=1.0)[0] == 0.0 and _run(squeeze=0.3)[0] == 0.0 and _run(quanta=0)[0] == 0.0
    made = [_run(rate=rate)[0] for rate in (0.5, 1.0, 4.0, 1000.0)]
    assert made[0] < made[1] < made[2] < made[3] <= level           # the rate always counts, never past the level
    slow = [_run(tau=tau)[0] for tau in (0.25e-6, 0.25e-3, 0.25)]
    assert 0.0 < slow[0] < slow[1] < slow[2]                         # a slow clock makes it slower, not nothing
    assert math.isclose(slow[0], level * 0.25e-6, rel_tol=1e-5)


def test_a_joint_collapses_when_it_holds_the_limit_times_what_it_was_made_with_squeezed_now_or_not():
    assert _run(holds=64800)[1] == 1.0 and _run(holds=64799)[1] == 0.0 and _run(holds=200000)[1] == 1.0
    assert _run(holds=10, fresh=0)[1] == 0.0


def test_a_volume_tears_only_when_too_long_and_each_of_its_three_pieces_would_hold_something():
    assert _run()[2] == 1.0 and _run(length=20.0)[2] == 0.0
    for pieces in ((0, 5, 5), (5, 0, 5), (5, 5, 0), (3, 0, 0)):
        assert _run(pieces=pieces)[2] == 0.0


def test_a_tear_keeps_every_quantum_flips_only_the_middle_and_each_piece_fills_its_block():
    quanta = np.random.default_rng(1).integers(0, 5000, (6, 6, 6)).astype(np.int32)
    phase = np.random.default_rng(2).uniform(-3.0, 3.0, (6, 6, 6))
    for axis in range(3):
        (low, low_phase), (middle, middle_phase), (high, high_phase) = tear_map(quanta, phase, axis)
        layers = np.moveaxis(quanta.astype(np.int64), axis, 0)
        assert int(low.sum()) == int(layers[0:2].sum()) and (low >= 0).all()
        assert int(-middle.sum()) == int(layers[2:4].sum()) and (middle <= 0).all()
        assert int(high.sum()) == int(layers[4:6].sum()) and (high >= 0).all()
        assert low_phase.dtype == np.float64
        assert np.array_equal(np.moveaxis(middle_phase, axis, 0)[0], np.moveaxis(phase, axis, 0)[2])
